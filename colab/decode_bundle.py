import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import samplerate
import soundfile as sf
from scipy.signal import chirp, correlate, fftconvolve
from scipy.stats import linregress, theilslopes

from generate_bundle import MLS_TAPS, generate_mls, gold_family, golay_pair, hadamard_matrix, reciprocal_mls

FMAX_HZ = 20000.0
DRIFT_QUALITY = "sinc_best"
OKNO_ZGRUBNE_S = 2.0
OKNO_DOKLADNE = 2
PROG_POMIARU = 0.1
PROG_RESZTY = 0.02
PROG_KWADRAT = 1.0


def read_wav_mono(path):
    data, fs = sf.read(path, always_2d=True)
    return fs, np.ascontiguousarray(data[:, 0], dtype=np.float64)


def bundle_mls(m):
    return generate_mls(MLS_TAPS[m], m)


def xcorr(y, ref, ir_len):
    c = fftconvolve(y, ref[::-1], mode="full")
    start = len(ref) - 1
    return c[start:start + ir_len]


def linear_chirp(fs, dur_s, f0=600.0, f1=6000.0, amp=0.6):
    n = int(round(dur_s * fs))
    t = np.arange(n, dtype=float) / fs
    sig = amp * chirp(t, f0=f0, t1=dur_s, f1=f1, method="linear", phi=-90)
    fade = int(min(n // 10, fs * 0.005))
    ramp = np.linspace(0.0, 1.0, fade)
    sig[:fade] *= ramp
    sig[-fade:] *= ramp[::-1]
    return sig


def pozycja_chirpu(y, wzorzec, srodek, okno):
    lo = max(0, srodek - okno)
    hi = min(len(y), srodek + okno + len(wzorzec))
    c = np.abs(correlate(y[lo:hi], wzorzec, mode="valid", method="fft"))
    i = int(np.argmax(c))
    poz = float(lo + i)
    if 0 < i < c.size - 1:
        mianownik = c[i - 1] - 2.0 * c[i] + c[i + 1]
        if mianownik != 0.0:
            poz += 0.5 * (c[i - 1] - c[i + 1]) / mianownik
    return poz, i in (0, c.size - 1)


def zmierz_dryft(y, meta, fs):
    syncs = [s for s in meta["segments"] if s["kind"] == "sync"]
    wzorzec = linear_chirp(fs, (syncs[0]["end"] - syncs[0]["start"]) / fs)
    t = np.array([s["start"] / fs for s in syncs])
    okno = int(round(OKNO_ZGRUBNE_S * fs))
    zgrubne, brzegi = [], []
    for s in syncs:
        poz, brzeg = pozycja_chirpu(y, wzorzec, s["start"], okno)
        zgrubne.append(poz - s["start"])
        brzegi.append(brzeg)
    zgrubne = np.array(zgrubne)
    a, b, _, _ = theilslopes(zgrubne, t)
    dokladne = []
    for k, s in enumerate(syncs):
        poz, brzeg = pozycja_chirpu(y, wzorzec, s["start"] + int(round(a * t[k] + b)), OKNO_DOKLADNE)
        dokladne.append(poz - s["start"])
        brzegi.append(brzeg)
    p = linregress(t, zgrubne)
    reszty = np.abs(zgrubne - (p.slope * t + p.intercept))
    czas = float(np.ptp(t))
    kwadrat = abs(float(np.polyfit((t - t[0]) / czas, zgrubne - zgrubne[0], 2)[0]))
    if any(brzegi):
        raise ValueError("Chirp synchronizacji na brzegu okna wyszukiwania")
    if np.any(np.abs(zgrubne - np.array(dokladne)) > PROG_POMIARU):
        raise ValueError("Rozbiezne polozenie chirpu w szerokim i waskim oknie")
    if reszty.max() > max(PROG_POMIARU, PROG_RESZTY * float(np.ptp(zgrubne))) or kwadrat > PROG_KWADRAT:
        raise ValueError("Przesuniecia chirpow nie narastaja liniowo")
    return p.slope / fs, zgrubne


def korekcja_dryftu(y, meta, fs):
    eps, _ = zmierz_dryft(y, meta, fs)
    y = samplerate.resample(y, 1.0 / (1.0 + eps), DRIFT_QUALITY)
    eps2, przesuniecia = zmierz_dryft(y, meta, fs)
    return y, int(round(float(przesuniecia[0]))), {"eps": eps, "residual_eps": eps2}


def fragment(y, seg, L, ir_len, pre):
    return y[seg["start"] - pre:seg["start"] + L + ir_len - 1]


def decode_solo_golay(y, meta, ir_len, pre=0):
    A, B = golay_pair(meta["golay_n"])
    L = len(A)
    seg = {s["name"]: s for s in meta["segments"]}
    irs = []
    for j in range(meta["speakers"]):
        ha = xcorr(fragment(y, seg[f"solo_golay_spk{j+1}_A"], L, ir_len, pre), A, ir_len + pre)
        hb = xcorr(fragment(y, seg[f"solo_golay_spk{j+1}_B"], L, ir_len, pre), B, ir_len + pre)
        irs.append((ha + hb) / (2.0 * L))
    return irs


def decode_solo_mls(y, meta, ir_len, pre=0):
    mls = bundle_mls(meta["mls_m"])
    L = len(mls)
    seg = {s["name"]: s for s in meta["segments"]}
    return [xcorr(fragment(y, seg[f"solo_mls_spk{j+1}"], L, ir_len, pre), mls, ir_len + pre) / L
            for j in range(meta["speakers"])]


def decode_golay_hadamard(y, meta, ir_len, pre=0):
    A, B = golay_pair(meta["golay_n"])
    L = len(A)
    k = meta["hadamard_size"]
    seg = {s["name"]: s for s in meta["segments"]}
    rows = []
    for i in range(k):
        ha = xcorr(fragment(y, seg[f"golay_had_row{i+1}_A"], L, ir_len, pre), A, ir_len + pre)
        hb = xcorr(fragment(y, seg[f"golay_had_row{i+1}_B"], L, ir_len, pre), B, ir_len + pre)
        rows.append((ha + hb) / (2.0 * L))
    decoded = hadamard_matrix(k).T @ np.array(rows)
    return list(decoded[:meta["speakers"]] / k)


def decode_mls_hadamard(y, meta, ir_len, pre=0):
    mls = bundle_mls(meta["mls_m"])
    L = len(mls)
    k = meta["hadamard_size"]
    seg = {s["name"]: s for s in meta["segments"]}
    rows = [xcorr(fragment(y, seg[f"mls_had_row{i+1}"], L, ir_len, pre), mls, ir_len + pre) / L
            for i in range(k)]
    decoded = hadamard_matrix(k).T @ np.array(rows)
    return list(decoded[:meta["speakers"]] / k)


def decode_mls_reciprocal(y, meta, ir_len, pre=0):
    mls = bundle_mls(meta["mls_m"])
    L = len(mls)
    seg = next(s for s in meta["segments"] if s["name"] == "mls_reciprocal_pair")
    yy = fragment(y, seg, L, ir_len, pre)
    return [xcorr(yy, code, ir_len + pre) / L for code in (mls, reciprocal_mls(mls))]


def decode_gold(y, meta, ir_len, pre=0):
    codes = gold_family(meta["gold_m"], meta["speakers"])
    L = len(codes[0])
    seg = next(s for s in meta["segments"] if s["name"] == "gold_all")
    yy = fragment(y, seg, L, ir_len, pre)
    return [xcorr(yy, code, ir_len + pre) / L for code in codes]


METHODS = {
    "solo_golay": ("Solo Golay", decode_solo_golay, "#1f77b4"),
    "golay_hadamard": ("Golay+Hadamard", decode_golay_hadamard, "#2ca02c"),
    "solo_mls": ("Solo MLS", decode_solo_mls, "#ff7f0e"),
    "mls_hadamard": ("MLS+Hadamard", decode_mls_hadamard, "#d62728"),
    "mls_reciprocal": ("MLS reciprocal", decode_mls_reciprocal, "#17becf"),
    "gold": ("Gold", decode_gold, "#9467bd"),
}

PLOT_PAIRS = [
    ("golay", ["solo_golay", "golay_hadamard"]),
    ("mls", ["solo_mls", "mls_hadamard"]),
    ("mls_reciprocal", ["solo_mls", "mls_reciprocal"]),
    ("gold", ["gold"]),
    ("gold_vs_solo", ["gold", "solo_mls", "solo_golay"]),
]


def poczatki_okien(solo_golay, zero, fs):
    starts = {}
    for j, ir in enumerate(solo_golay, 1):
        peak = int(np.argmax(np.abs(ir[:zero + int(round(.005 * fs))])))
        starts[j] = peak - int(round(.001 * fs))
    return starts


def compute_fft_mag_db(ir, fs, fmax=None):
    mag_db = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(ir)), 1e-12))
    f = np.fft.rfftfreq(len(ir), d=1.0 / fs)
    if fmax is not None:
        m = f <= float(fmax)
        f, mag_db = f[m], mag_db[m]
    return f, mag_db


def plot_methods_per_speaker(keys, irs, title, out_path, fs, fmin):
    n_spk = len(irs[keys[0]])
    fig, axes = plt.subplots(n_spk, 1, figsize=(14, 3 + 3 * n_spk), squeeze=False, sharex=True)
    fig.suptitle(title, fontsize=14)
    axes = [row[0] for row in axes]
    for j, ax in enumerate(axes):
        for key in keys:
            label, _, color = METHODS[key]
            f, mag_db = compute_fft_mag_db(irs[key][j], fs)
            visible = (f >= fmin) & (f <= min(FMAX_HZ, fs / 2))
            ax.plot(f[visible], mag_db[visible], color=color, lw=0.7, label=label)
        ax.set_ylabel(f"spk{j + 1}  |H(f)| [dB]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[-1].set_xlabel("f [Hz]")
    axes[-1].set_xlim(fmin, FMAX_HZ)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def excitation_scale(bundle_dir, meta, override):
    if override is not None:
        return override
    if "excitation_gain" in meta:
        return meta["excitation_gain"]
    wav = os.path.join(bundle_dir, "bundle.wav")
    if os.path.exists(wav):
        return float(np.max(np.abs(sf.read(wav)[0])))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--recording", required=True)
    p.add_argument("--ir-len", type=int, default=8192)
    p.add_argument("--pre-roll", type=int, default=256)
    p.add_argument("--out-dir")
    p.add_argument("--all-methods", action="store_true")
    p.add_argument("--no-drift", action="store_true")
    p.add_argument("--excitation-gain", type=float)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--fmin-plot", type=float, default=60.0)
    args = p.parse_args()

    bundle_dir = os.path.abspath(args.bundle_dir)
    with open(os.path.join(bundle_dir, "manifest.json"), encoding="utf-8") as f:
        meta = json.load(f)
    fs = meta["fs"]
    _, y = read_wav_mono(args.recording)
    gain = excitation_scale(bundle_dir, meta, args.excitation_gain)

    offset, drift = 0, None
    if not args.no_drift:
        y, offset, drift = korekcja_dryftu(y, meta, fs)
    shifted = dict(meta, segments=[dict(s, start=s["start"] + offset, end=s["end"] + offset)
                                   for s in meta["segments"]])

    names = {s["name"] for s in meta["segments"]}
    decoded = {}
    for key, (label, decode, _) in METHODS.items():
        if not args.all_methods and key not in ("solo_golay", "golay_hadamard"):
            continue
        if key == "mls_reciprocal" and "mls_reciprocal_pair" not in names:
            continue
        print(f"{label}...")
        decoded[key] = [ir / (gain or 1.0) for ir in decode(y, shifted, args.ir_len, args.pre_roll)]

    out_dir = os.path.abspath(args.out_dir or os.path.join(bundle_dir, "decoded"))
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for key, irs in decoded.items():
        for j, ir in enumerate(irs, 1):
            files.append(f"decoded_{key}_spk{j}.npy")
            np.save(os.path.join(out_dir, files[-1]), ir)
    run = {"fs": fs, "recording": os.path.relpath(args.recording).replace(os.sep, "/"), "bundle_manifest": meta,
           "sync_mode": "none" if args.no_drift else "full", "offset_samples": offset, "drift": drift,
           "zero_index": args.pre_roll, "ir_len_after_zero": args.ir_len,
           "excitation_gain": gain, "files": files}
    with open(os.path.join(out_dir, "decode_run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
    print(f"Wyniki: {out_dir}")

    if args.plot:
        n = int(round(0.100 * fs))
        starts = poczatki_okien(decoded["solo_golay"], args.pre_roll, fs)
        windows = {key: [ir[starts[j]:starts[j] + n] for j, ir in enumerate(irs, 1)]
                   for key, irs in decoded.items()}
        n_spk = meta["speakers"]
        for suffix, keys in PLOT_PAIRS:
            if all(k in windows for k in keys):
                title = " / ".join(METHODS[k][0] for k in keys) + " — okno analizy: 100 ms"
                plot_methods_per_speaker(keys, windows, title,
                                         os.path.join(out_dir, f"fft_{n_spk}spk_{suffix}.png"), fs, args.fmin_plot)
        plot_methods_per_speaker(list(windows), windows, "Wszystkie metody — okno analizy: 100 ms",
                                 os.path.join(out_dir, "fft_all_methods.png"), fs, args.fmin_plot)


if __name__ == "__main__":
    main()
