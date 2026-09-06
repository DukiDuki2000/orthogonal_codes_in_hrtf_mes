import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, chirp, correlate
from scipy.stats import theilslopes
import samplerate
import matplotlib.pyplot as plt

FMAX_HZ = 20000.0
FMIN_PLOT_HZ = 60.0
FFT_SMOOTH_FRACTION = 3
FFT_SMOOTH_FMIN = 20.0
DRIFT_QUALITY = "sinc_best"

def read_wav_mono(path):
    data, fs = sf.read(path, always_2d=False)
    data = np.asarray(data, dtype=np.float64)

    if data.ndim > 1:
        data = data[:, 0]

    return fs, data



def golay_pair(n):
    A, B = np.array([1.0]), np.array([1.0])
    for _ in range(n):
        A, B = np.concatenate([A, B]), np.concatenate([A, -B])
    return A, B

def generate_mls(taps, m):
    reg = [1] * m
    seq = []
    for _ in range(2**m - 1):
        seq.append(1.0 if reg[-1] else -1.0)
        fb = 0
        for t in taps:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]
    return np.array(seq)

def bundle_mls(m):
    return generate_mls([17, 14], m)

def reciprocal_mls(seq):
    if len(seq) <= 1:
        return np.array(seq, copy=True)
    return np.concatenate([seq[:1], seq[:0:-1]])

def gold_family(m, n_codes):
    mls1 = generate_mls([16, 14, 13, 11], m)
    mls2 = generate_mls([16, 15, 13, 4], m)
    codes = [mls1, mls2]
    for k in range(len(mls1)):
        if len(codes) >= n_codes:
            break
        codes.append(mls1 * np.roll(mls2, k))
    return codes[:n_codes]


def hadamard_matrix(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

def xcorr(y, ref, ir_len):
    c = fftconvolve(y, ref[::-1], mode='full')
    start = len(ref) - 1
    return c[start:start + ir_len]

def linear_chirp(fs, dur_s, f0=600.0, f1=6000.0, amp=0.6):
    n = int(round(dur_s * fs))
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n, dtype=float) / fs
    sig = amp * chirp(t, f0=f0, t1=max(dur_s, 1e-12), f1=f1, method="linear", phi=-90)
    fade = int(min(n // 10, fs * 0.005))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]
    return sig


def _pozycja_chirpu(y, expected, ref, margin):
    lo = max(0, expected - margin)
    okno = y[lo:min(len(y), expected + margin + len(ref))]
    if okno.size < ref.size:
        return None
    corr = np.abs(correlate(okno, ref, mode="valid", method="fft"))
    return lo + int(np.argmax(corr)) - expected


def measure_drift(y, meta, fs):

    syncs = [s for s in meta["segments"] if s["kind"] == "sync"]
    if len(syncs) < 2:
        return np.zeros(0), np.zeros(0)

    ref = linear_chirp(fs, (syncs[0]["end"] - syncs[0]["start"]) / fs)
    margin = 2 * int(fs)
    zgrubny = _pozycja_chirpu(y, syncs[0]["start"], ref, margin)
    if zgrubny is None:
        return np.zeros(0), np.zeros(0)

    pad = 200
    wytnij = lambda s: y[s["start"] + zgrubny - pad:
                         s["start"] + zgrubny + len(ref) + pad]
    wzor = wytnij(syncs[0])

    times, shifts = [], []
    for sync in syncs:
        frag = wytnij(sync)
        if frag.size != wzor.size:
            continue
        corr = np.abs(correlate(frag, wzor, mode="same", method="fft"))
        i = int(np.argmax(corr))
        d = float(i - corr.size // 2)
        if 0 < i < corr.size - 1:
            kw, lin, _ = np.polyfit([-1.0, 0.0, 1.0], corr[i - 1:i + 2], 2)
            if kw != 0.0:
                d -= lin / (2.0 * kw)
        times.append(sync["start"] / fs)
        shifts.append(d)
    return np.asarray(times), np.asarray(shifts)


def correct_drift(y, meta, fs, quality=DRIFT_QUALITY):
    times, shifts = measure_drift(y, meta, fs)
    if times.size < 2:
        return y, 0

    a, b, _, _ = theilslopes(shifts, times)
    eps = a / fs
    y2 = samplerate.resample(np.asarray(y, dtype=np.float64), 1.0 / (1.0 + eps), quality)

    _, shifts2 = measure_drift(y2, meta, fs)
    syncs = [s for s in meta["segments"] if s["kind"] == "sync"]
    ref = linear_chirp(fs, (syncs[0]["end"] - syncs[0]["start"]) / fs)
    offset = _pozycja_chirpu(y2, syncs[0]["start"], ref, 2 * int(fs))
    return y2, int(offset or 0)


def decode_solo_golay(y, meta, ir_len):
    n_spk = meta["speakers"]
    A, B = golay_pair(meta["golay_n"])
    L = meta["golay_L"]
    segs = meta["segments"]

    irs = []
    for j in range(n_spk):
        name_a = f"solo_golay_spk{j+1}_A"
        name_b = f"solo_golay_spk{j+1}_B"
        seg_a = next(s for s in segs if s["name"] == name_a)
        seg_b = next(s for s in segs if s["name"] == name_b)
        ya = y[seg_a["start"]:seg_a["start"] + L + ir_len]
        yb = y[seg_b["start"]:seg_b["start"] + L + ir_len]
        ha = xcorr(ya, A, ir_len)
        hb = xcorr(yb, B, ir_len)
        h = (ha + hb) / (2.0 * L)
        irs.append(h)
    return irs


def decode_solo_mls(y, meta, ir_len):
    n_spk = meta["speakers"]
    mls_m = meta["mls_m"]
    mls = bundle_mls(mls_m)
    L = meta["mls_L"]
    segs = meta["segments"]

    irs = []
    for j in range(n_spk):
        name = f"solo_mls_spk{j+1}"
        seg = next(s for s in segs if s["name"] == name)
        yy = y[seg["start"]:seg["start"] + L + ir_len]
        h = xcorr(yy, mls, ir_len) / L
        irs.append(h)
    return irs


def decode_golay_hadamard(y, meta, ir_len):
    n_spk = meta["speakers"]
    h_size = meta["hadamard_size"]
    A, B = golay_pair(meta["golay_n"])
    L = meta["golay_L"]
    H = hadamard_matrix(h_size)
    segs = meta["segments"]
    row_irs = []
    for i in range(h_size):
        name_a = f"golay_had_row{i+1}_A"
        name_b = f"golay_had_row{i+1}_B"
        seg_a = next(s for s in segs if s["name"] == name_a)
        seg_b = next(s for s in segs if s["name"] == name_b)
        ya = y[seg_a["start"]:seg_a["start"] + L + ir_len]
        yb = y[seg_b["start"]:seg_b["start"] + L + ir_len]
        ha = xcorr(ya, A, ir_len)
        hb = xcorr(yb, B, ir_len)
        row_irs.append(ha + hb)
    decoded = H.T @ np.array(row_irs)
    return list(decoded[:n_spk] / (2.0 * L * h_size))


def decode_mls_hadamard(y, meta, ir_len):
    n_spk = meta["speakers"]
    h_size = meta["hadamard_size"]
    mls_m = meta["mls_m"]
    mls = bundle_mls(mls_m)
    L = meta["mls_L"]
    H = hadamard_matrix(h_size)
    segs = meta["segments"]
    row_irs = []
    for i in range(h_size):
        name = f"mls_had_row{i+1}"
        seg = next(s for s in segs if s["name"] == name)
        yy = y[seg["start"]:seg["start"] + L + ir_len]
        h = xcorr(yy, mls, ir_len)
        row_irs.append(h)
    decoded = H.T @ np.array(row_irs)
    return list(decoded[:n_spk] / (L * h_size))


def decode_mls_reciprocal(y, meta, ir_len):
    mls_m = meta["mls_m"]
    mls = bundle_mls(mls_m)
    codes = [mls, reciprocal_mls(mls)]
    L = meta["mls_L"]
    segs = meta["segments"]
    seg = next(s for s in segs if s["name"] == "mls_reciprocal_pair")
    yy = y[seg["start"]:seg["start"] + L + ir_len]
    irs = []
    for code in codes:
        irs.append(xcorr(yy, code, ir_len) / L)
    return irs


def decode_gold(y, meta, ir_len):
    n_spk = meta["speakers"]
    codes = gold_family(meta["gold_m"], n_spk)
    L = meta["gold_L"]
    segs = meta["segments"]

    seg = next(s for s in segs if s["name"] == "gold_all")
    yy = y[seg["start"]:seg["start"] + L + ir_len]

    irs = []
    for j in range(n_spk):
        h = xcorr(yy, codes[j], ir_len) / L
        irs.append(h)
    return irs


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    decode: object
    color: str
    always: bool = False


METHODS = [
    Method("solo_golay",     "Solo Golay",     decode_solo_golay,     "#1f77b4", always=True),
    Method("golay_hadamard", "Golay+Hadamard", decode_golay_hadamard, "#2ca02c", always=True),
    Method("solo_mls",       "Solo MLS",       decode_solo_mls,       "#ff7f0e"),
    Method("mls_hadamard",   "MLS+Hadamard",   decode_mls_hadamard,   "#d62728"),
    Method("mls_reciprocal", "MLS reciprocal", decode_mls_reciprocal, "#17becf"),
    Method("gold",           "Gold",           decode_gold,           "#9467bd"),
]

METHOD = {m.key: m for m in METHODS}

PLOT_PAIRS = [
    ("golay",          ["solo_golay", "golay_hadamard"]),
    ("mls",            ["solo_mls", "mls_hadamard"]),
    ("mls_reciprocal", ["solo_mls", "mls_reciprocal"]),
    ("gold",           ["gold"]),
    ("gold_vs_solo",   ["gold", "solo_mls", "solo_golay"]),
]



def compute_fft_mag_db(ir, fs, fmax=None):
    x = np.asarray(ir, dtype=float)
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    if fmax is not None:
        m = f <= float(fmax)
        f, mag_db = f[m], mag_db[m]
    return f, mag_db


def smooth_fractional_octave(f_hz, mag_db, fraction=3, fmin_hz=20.0):
    f = np.asarray(f_hz, dtype=float)
    y = np.asarray(mag_db, dtype=float)
    if f.size == 0:
        return y.copy()
    out = y.copy()
    m_pos = f > 0
    if not np.any(m_pos):
        return out
    m_start = f >= fmin_hz
    idx0 = int(np.argmax(m_pos & m_start)) if np.any(m_pos & m_start) else int(np.argmax(m_pos))
    f2, y2 = f[idx0:], y[idx0:]
    if f2.size == 0:
        return out
    logf = np.log2(f2)
    half_bw = 1.0 / (2.0 * fraction)
    for i in range(logf.size):
        lo, hi = logf[i] - half_bw, logf[i] + half_bw
        ilo = int(np.searchsorted(logf, lo, side="left"))
        ihi = int(np.searchsorted(logf, hi, side="right"))
        if ihi > ilo:
            out[idx0 + i] = float(np.mean(y2[ilo:ihi]))
    return out


def _fft_curve(ir, fs):
    f, mag_db = compute_fft_mag_db(ir, fs, fmax=FMAX_HZ)
    mag_sm = smooth_fractional_octave(f, mag_db, fraction=FFT_SMOOTH_FRACTION,
                                      fmin_hz=FFT_SMOOTH_FMIN)
    return f, mag_db, mag_sm


def _normalize_at(f, mag_db, mag_sm, norm_hz):
    if not norm_hz or f.size == 0:
        return mag_db, mag_sm
    idx = int(np.argmin(np.abs(f - float(norm_hz))))
    offset = float(mag_sm[idx])
    return mag_db - offset, mag_sm - offset


def _fft_figure(n_spk, title):
    fig, axes = plt.subplots(n_spk, 1, figsize=(14, 3 + 3 * n_spk),
                             squeeze=False, sharex=True)
    fig.suptitle(title, fontsize=14)
    return [row[0] for row in axes]


def _save_fft_figure(axes, out_path):
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[-1].set_xlabel("f [Hz]")
    axes[-1].set_xlim(FMIN_PLOT_HZ, FMAX_HZ)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_methods_per_speaker(keys, all_decoded, title, out_path, n_spk, fs, norm_hz=None):
    axes = _fft_figure(n_spk, title)
    for j, ax in enumerate(axes):
        for key in keys:
            method = METHOD[key]
            f, mag_db, mag_sm = _fft_curve(all_decoded[key][j], fs)
            mag_db, mag_sm = _normalize_at(f, mag_db, mag_sm, norm_hz)
            ax.plot(f, mag_db, color=method.color, alpha=0.15, lw=0.5)
            ax.plot(f, mag_sm, color=method.color, lw=1.4, label=method.label)
        if norm_hz:
            ax.axvline(float(norm_hz), color="0.4", ls="--", lw=1.0)
            ax.axhline(0.0, color="0.4", ls=":", lw=1.0)
            ax.set_ylabel(f"spk{j + 1}  |H(f)| [dB] (0 dB @ {int(norm_hz)} Hz)")
        else:
            ax.set_ylabel(f"spk{j + 1}  |H(f)| [dB]")

    _save_fft_figure(axes, out_path)
    print(f"Wykres: {out_path}")


def main():
    global FMIN_PLOT_HZ
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", required=True, help="Folder z manifest.json")
    p.add_argument("--recording", required=True, help="Ścieżka do mono WAV nagrania")
    p.add_argument("--ir-len", type=int, default=4096, help="Długość IR [próbki]")
    p.add_argument("--plot", action="store_true", help="Generuj wykresy FFT")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--all-methods", action="store_true")
    p.add_argument("--fmin-plot", type=float, default=FMIN_PLOT_HZ)
    p.add_argument("--norm-hz", type=float, default=None,
                   help="Dodatkowe wykresy z wyrownaniem poziomu do zadanej f [Hz], np. 3000")
    args = p.parse_args()

    FMIN_PLOT_HZ = args.fmin_plot

    bundle_dir = os.path.abspath(args.bundle_dir)
    with open(os.path.join(bundle_dir, "manifest.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_spk = meta["speakers"]
    fs = meta["fs"]
    ir_len = args.ir_len

    fs_rec, y = read_wav_mono(args.recording)

    y, offset = correct_drift(y, meta, fs, quality=args.drift_quality)
    for seg in meta["segments"]:
        seg["start"] += offset
        seg["end"] += offset

    has_mls_reciprocal = any(s["name"] == "mls_reciprocal_pair" for s in meta["segments"])
    all_decoded = {}
    for method in METHODS:
        if not method.always and not args.all_methods:
            continue
        if method.key == "mls_reciprocal" and not has_mls_reciprocal:
            continue
        print(f"{method.label}...")
        all_decoded[method.key] = method.decode(y, meta, ir_len)

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(bundle_dir, "decoded")
    os.makedirs(out_dir, exist_ok=True)

    for key, irs in all_decoded.items():
        for j, ir in enumerate(irs):
            np.save(os.path.join(out_dir, f"decoded_{key}_spk{j + 1}.npy"), ir)

    if args.plot:
        okt = f"1/{FFT_SMOOTH_FRACTION} okt."
        norm_variants = [(None, "")]
        if args.norm_hz:
            norm_variants.append((args.norm_hz, f"_norm{int(args.norm_hz)}"))

        for norm_hz, tag in norm_variants:
            norm_txt = f", 0 dB @ {int(norm_hz)} Hz" if norm_hz else ""
            for suffix, keys in PLOT_PAIRS:
                if not all(k in all_decoded for k in keys):
                    continue
                label = " vs ".join(METHOD[k].label for k in keys)
                plot_methods_per_speaker(
                    keys, all_decoded,
                    title=f"{label} ({n_spk} spk) - charakterystyki "
                          f"czestotliwosciowe ({okt}{norm_txt})",
                    out_path=os.path.join(out_dir, f"fft_{n_spk}spk_{suffix}{tag}.png"),
                    n_spk=n_spk, fs=fs, norm_hz=norm_hz)

            present = [m.key for m in METHODS if m.key in all_decoded]
            if present:
                plot_methods_per_speaker(
                    present, all_decoded,
                    title=f"Charakterystyki czestotliwosciowe - {len(present)} metod "
                          f"({n_spk} spk, {okt}{norm_txt})",
                    out_path=os.path.join(out_dir, f"fft_all_methods{tag}.png"),
                    n_spk=n_spk, fs=fs, norm_hz=norm_hz)


if __name__ == "__main__":
    main()
