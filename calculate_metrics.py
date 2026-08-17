from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decode_bundle

DECODED_RE = re.compile(r"^decoded_(?P<method>.+)_spk(?P<speaker>\d+)\.npy$")

MAX_HZ = 20000.0
C80_MS = 80.0
TAIL_FRACTION = 0.10
TAIL_MS = None
NOISE_WINDOW_S = 2.0
NOISE_SEGMENT = "cisza_koncowa_noise_floor"
OUT_NAME = "tabela.csv"
MAX_HZ_EFF = MAX_HZ if MAX_HZ > 0 else None
MAX_LAG_MS = 3.0      
FULL_LOW_HZ = 1500.0

WYMAGANY_SEGMENT = {
    "solo_golay": "solo_golay_A",
    "solo_mls": "solo_mls",
    "golay_hadamard": "golay_hadamard_A",
    "mls_hadamard": "mls_hadamard",
    "mls_reciprocal": "mls_reciprocal",
    "gold": "gold",
}

METHOD_LABELS = {
    "golay_hadamard": "Golay+Had.",
    "solo_golay": "Solo Golay",
    "mls_hadamard": "MLS+Had.",
    "solo_mls": "Solo MLS",
    "mls_reciprocal": "MLS recip.",
    "gold": "Gold",
}

COMPARISON_SPECS = [
    ("golay_hadamard", "solo_golay", "GH-Solo"),
    ("mls_hadamard", "solo_mls", "MLSH-Solo"),
    ("mls_reciprocal", "gold", "MLSrec-Gold"),
]

@dataclass(frozen=True)
class Band:
    label: str
    center_hz: float | None
    low_hz: float | None
    high_hz: float | None


def parse_bands(text: str, fs: float, max_hz: float | None) -> list[Band]:
    bands = [Band("full", None, None, None)]
    upper = fs / 2.0 if max_hz is None else min(fs / 2.0, float(max_hz))
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        center = float(part)
        low = center / math.sqrt(2.0)
        if low >= upper:
            continue
        bands.append(Band(str(int(center)), center, low, min(center * math.sqrt(2.0), upper)))
    return bands


def band_limited(ir: np.ndarray, fs: float, band: Band) -> np.ndarray:
    x = np.asarray(ir, dtype=float)
    if band.center_hz is None or x.size == 0:
        return x.copy()
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spectrum[~((freqs >= float(band.low_hz)) & (freqs <= float(band.high_hz)))] = 0.0
    return np.fft.irfft(spectrum, n=x.size)


def band_edges(band: Band, max_hz: float | None) -> tuple[float, float]:
    if band.center_hz is None:
        return FULL_LOW_HZ, (max_hz if max_hz else 20000.0)
    return float(band.low_hz), float(band.high_hz)


def load_decoded_ir(decoded_dir: str) -> dict[str, dict[int, np.ndarray]]:
    methods: dict[str, dict[int, np.ndarray]] = {}
    for filename in sorted(os.listdir(decoded_dir)):
        match = DECODED_RE.match(filename)
        if not match:
            continue
        ir = np.asarray(np.load(os.path.join(decoded_dir, filename)), dtype=float)
        methods.setdefault(match.group("method"), {})[int(match.group("speaker"))] = ir
    return methods


def detect_ir_len(decoded_dirs: list[str]) -> tuple[int, set[int]]:
    lengths: set[int] = set()
    for decoded_dir in decoded_dirs:
        for name in os.listdir(decoded_dir):
            if DECODED_RE.match(name):
                lengths.add(int(np.load(os.path.join(decoded_dir, name), mmap_mode="r").shape[0]))
    if not lengths:
        raise FileNotFoundError(f"Brak plików")
    return max(lengths), lengths


def db10(value: float) -> float:
    return 10.0 * math.log10(value) if value > 0.0 else float("-inf")


def select_tail_samples(n_samples: int, fs: float, tail_fraction: float,
                        tail_ms: float | None) -> int:
    if n_samples < 4:
        return 0
    requested = int(round(tail_ms * fs / 1000.0)) if tail_ms is not None \
        else int(round(n_samples * tail_fraction))
    return min(max(8, requested), max(1, n_samples // 2))


def signal_power_for_snr(ir: np.ndarray, fs: float, tail_fraction: float,
                         tail_ms: float | None) -> float:
    tail_n = select_tail_samples(ir.size, fs, tail_fraction, tail_ms)
    signal = ir[:-tail_n] if 0 < tail_n < ir.size else ir
    return float(np.mean(signal * signal))


def c80_db(ir: np.ndarray, fs: float, c80_ms: float) -> float:
    split = int(round(c80_ms * fs / 1000.0))
    if split <= 0 or split >= ir.size:
        return float("nan")
    return db10(float(np.sum(ir[:split] ** 2)) / max(float(np.sum(ir[split:] ** 2)), 1e-30))


def quietest_window(y: np.ndarray, window_samples: int) -> tuple[np.ndarray, int]:
    if y.size < window_samples:
        return y.copy(), 0
    cum = np.concatenate(([0.0], np.cumsum(y.astype(np.float64) ** 2)))
    sums = cum[window_samples:] - cum[:-window_samples]
    start = int(np.argmin(sums))
    return y[start:start + window_samples].copy(), start


def load_noise_segment(recording_path: str, manifest_path: str, segment_name: str,
                       read_wav_mono) -> tuple[np.ndarray, int]:
    fs, recording = read_wav_mono(recording_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    segment = next((s for s in manifest["segments"] if s["name"] == segment_name), None)
    start, end = int(segment["start"]), int(segment["end"])
    return np.asarray(recording[start:end], dtype=float), fs


def load_noise(args, read_wav_mono) -> tuple[np.ndarray, int, str]:
    if args.noise_wav:
        fs, y = read_wav_mono(args.noise_wav)
        chunk, start = quietest_window(np.asarray(y, dtype=float),
                                       int(round(NOISE_WINDOW_S * fs)))
        return chunk, fs, (f"najcichsze {NOISE_WINDOW_S:.2f} s z "
                           f"{os.path.basename(args.noise_wav)} (od {start / fs:.2f} s)")
    noise, fs = load_noise_segment(args.calibration_recording, args.calibration_manifest,
                                   NOISE_SEGMENT, read_wav_mono)
    return noise, fs, (f"segment {NOISE_SEGMENT} z "
                       f"{os.path.basename(args.calibration_recording)}")


def required_recording_len(meta: dict, ir_len: int) -> int:
    kinds = {"solo_golay_A", "solo_golay_B", "solo_mls", "golay_hadamard_A",
             "golay_hadamard_B", "mls_hadamard", "mls_reciprocal_pair", "gold"}
    golay_len = int(meta["golay_L"])
    ends = [int(s["end"]) for s in meta["segments"]]
    ends += [int(s["start"]) + golay_len + ir_len
             for s in meta["segments"] if s.get("kind") in kinds]
    return max(ends)


def decode_noise(args, ir_len: int) -> tuple[dict[str, dict[int, np.ndarray]], float, str]:
    with open(args.bundle_manifest, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    fs = float(meta["fs"])

    noise, fs_noise, source = load_noise(args, decode_bundle.read_wav_mono)
    noise = noise - float(np.mean(noise))
    n_needed = required_recording_len(meta, ir_len)
    tiled = np.tile(noise, int(math.ceil(n_needed / noise.size)))[:n_needed].copy()

    obecne = {s["kind"] for s in meta["segments"]}
    decoded = {}
    for method in decode_bundle.METHODS:
        if WYMAGANY_SEGMENT[method.key] not in obecne:
            continue
        irs = method.decode(tiled, deepcopy(meta), ir_len)
        decoded[method.key] = {i + 1: ir for i, ir in enumerate(irs)}
    return decoded, fs, source

def shift_signal(x: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return x.copy()
    y = np.zeros_like(x)
    if shift > 0:
        y[shift:] = x[:-shift]
    else:
        y[:shift] = x[-shift:]
    return y


def align_to_reference(reference: np.ndarray, test: np.ndarray, fs: float) -> np.ndarray:
    n = min(reference.size, test.size)
    ref, tst = reference[:n], test[:n]
    max_lag = min(int(round(MAX_LAG_MS * fs / 1000.0)), max(n - 1, 0))
    ref_energy = float(np.dot(ref, ref))
    best_lag, best_score = 0, -float("inf")
    for lag in range(-max_lag, max_lag + 1):
        shifted = shift_signal(tst, lag)
        denom = math.sqrt(max(ref_energy * float(np.dot(shifted, shifted)), 1e-30))
        score = float(np.dot(ref, shifted)) / denom
        if score > best_score:
            best_score, best_lag = score, lag
    return shift_signal(test, best_lag)


def smooth_oct(f: np.ndarray, y: np.ndarray, fraction: int = 3, fmin: float = 20.0) -> np.ndarray:
    f, y = np.asarray(f, dtype=float), np.asarray(y, dtype=float)
    out = y.copy()
    m = f >= fmin
    if not np.any(m):
        return out
    i0 = int(np.argmax(m))
    lf = np.log2(f[i0:])
    hb = 1.0 / (2.0 * fraction)
    for k in range(lf.size):
        lo = int(np.searchsorted(lf, lf[k] - hb, side="left"))
        hi = int(np.searchsorted(lf, lf[k] + hb, side="right"))
        if hi > lo:
            out[i0 + k] = float(np.mean(y[i0 + lo:i0 + hi]))
    return out


def complex_band_correlation(h_ref: np.ndarray, h_test: np.ndarray, freqs: np.ndarray,
                             low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return float("nan")
    a, b = h_ref[mask], h_test[mask]
    denom = math.sqrt(max(float(np.sum(np.abs(a) ** 2)) * float(np.sum(np.abs(b) ** 2)), 1e-30))
    return abs(complex(np.sum(a * np.conj(b)) / denom))


def pair_metrics(reference: np.ndarray, test: np.ndarray, fs: float, bands: list[Band],
                 max_hz: float | None) -> dict[str, dict[str, float]]:
    n = min(reference.size, test.size)
    reference, test = reference[:n], test[:n]
    test_aligned = align_to_reference(reference, test, fs)[:n]

    nfft = int(2 ** math.ceil(math.log2(max(n, 2) * 4)))
    h_ref = np.fft.rfft(reference, n=nfft)
    h_test = np.fft.rfft(test_aligned, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    win = np.hanning(n)
    ref_mag = np.abs(np.fft.rfft(reference * win))
    test_mag = np.abs(np.fft.rfft(test * win))
    freqs_h = np.fft.rfftfreq(n, d=1.0 / fs)
    dmag_sm = (smooth_oct(freqs_h, 20.0 * np.log10(np.maximum(test_mag, 1e-12)))
               - smooth_oct(freqs_h, 20.0 * np.log10(np.maximum(ref_mag, 1e-12))))

    out: dict[str, dict[str, float]] = {}
    for band in bands:
        ref_b = band_limited(reference, fs, band)
        err = band_limited(test_aligned, fs, band) - ref_b
        rmse_rel = (math.sqrt(float(np.mean(err * err)))
                    / max(math.sqrt(float(np.mean(ref_b * ref_b))), 1e-30))

        low, high = band_edges(band, max_hz)
        hmask = (freqs_h >= low) & (freqs_h <= high)
        if np.any(hmask):
            seg = dmag_sm[hmask]
            mag_diff = float(seg[int(np.argmax(np.abs(seg)))])
        else:
            mag_diff = float("nan")

        out[band.label] = {
            "rmse": 20.0 * math.log10(max(rmse_rel, 1e-30)),
            "corr": complex_band_correlation(h_ref, h_test, freqs, low, high),
            "dmag": mag_diff,
        }
    return out

class Row(NamedTuple):
    metoda: str
    glosnik: str
    parametr: str
    wartosci: list
    miejsca: int

PAIR_ROWS = [("|rc|", "corr", 3), ("dMag [dB]", "dmag", 2), ("RMSE [dB]", "rmse", 2)]


def format_cell(value, decimals: int) -> str:
    if value is None:
        return "-"
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return "-"
    return f"{v:.{decimals}f}".replace(".", ",")


def subtract(test_vals, ref_vals) -> list:
    return [None if t is None or r is None else t - r for t, r in zip(test_vals, ref_vals)]


def build_table(values: dict, decoded: dict[str, dict[int, np.ndarray]], fs: float,
                bands: list[Band], max_hz: float | None) -> tuple[list[str], list[Row]]:
    labels = [b.label for b in bands]

    def seria(method, speaker, key):
        return [values.get((method, speaker, b), {}).get(key) for b in labels]

    rows = []
    for method in (m for m in METHOD_LABELS if m in decoded):
        for speaker in sorted(decoded[method]):
            rows.append(Row(METHOD_LABELS[method], str(speaker), "SNR [dB]",
                            seria(method, speaker, "snr"), 2))
            rows.append(Row("", "", "C80 [dB]", seria(method, speaker, "c80"), 2))

    for test_m, ref_m, label in COMPARISON_SPECS:
        if test_m not in decoded or ref_m not in decoded:
            continue
        for speaker in sorted(set(decoded[test_m]) & set(decoded[ref_m])):
            pm = pair_metrics(decoded[ref_m][speaker], decoded[test_m][speaker],
                              fs, bands, max_hz)
            rows.append(Row(f"d {label}", f"{speaker}-{speaker}", "dSNR [dB]",
                            subtract(seria(test_m, speaker, "snr"),
                                     seria(ref_m, speaker, "snr")), 2))
            rows.append(Row("", "", "dC80 [dB]",
                            subtract(seria(test_m, speaker, "c80"),
                                     seria(ref_m, speaker, "c80")), 2))
            for parametr, key, miejsca in PAIR_ROWS:
                rows.append(Row("", "", parametr, [pm[b][key] for b in labels], miejsca))

    header = ["Metoda", "Gl.", "Parametr"] + \
             ["pelne" if b == "full" else f"{b} Hz" for b in labels]
    return header, rows


def write_table_csv(path: str, header: list[str], rows: list[Row]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.metoda, row.glosnik, row.parametr] +
                            [format_cell(v, row.miejsca) for v in row.wartosci])



def analyze_dir(decoded_dir: str, args, bands: list[Band], fs: float,
                noise_decoded: dict[str, dict[int, np.ndarray]]) -> None:
    decoded_dir = os.path.abspath(decoded_dir)
    decoded = load_decoded_ir(decoded_dir)
    if not decoded:
        raise FileNotFoundError(f"Brak plikow")

    values: dict[tuple[str, int, str], dict[str, float | None]] = {}
    missing_noise = []
    for method in sorted(decoded):
        for speaker in sorted(decoded[method]):
            full_ir = decoded[method][speaker]
            noise_ir = noise_decoded.get(method, {}).get(speaker)
            if noise_ir is None:
                missing_noise.append(f"{method} spk{speaker}")
            for band in bands:
                ir_b = band_limited(full_ir, fs, band)
                snr = None
                if noise_ir is not None:
                    noise_b = band_limited(np.asarray(noise_ir)[:full_ir.size], fs, band)
                    snr = db10(signal_power_for_snr(ir_b, fs, TAIL_FRACTION, TAIL_MS)
                               / max(float(np.mean(noise_b * noise_b)), 1e-30))
                values[(method, speaker, band.label)] = {
                    "snr": snr,
                    "c80": c80_db(ir_b, fs, C80_MS),
                }

    header, rows = build_table(values, decoded, fs, bands, MAX_HZ_EFF)
    out_path = os.path.join(decoded_dir, OUT_NAME)
    write_table_csv(out_path, header, rows)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-dir", action="append", required=True,help="Katalog z plikami decoded.npy.")
    p.add_argument("--bundle-manifest", required=True)
    p.add_argument("--bands", default="125,250,500,1000,2000,4000,8000,16000")

    src = p.add_argument_group("Zrodlo ciszy")
    src.add_argument("--noise-wav", help="(2. sesja).")
    src.add_argument("--calibration-recording", help="(1. sesja).")
    src.add_argument("--calibration-manifest", help="(1. sesja).")

    args = p.parse_args()

    if bool(args.noise_wav) == bool(args.calibration_recording and args.calibration_manifest):
        p.error("error")

    ir_len, dlugosci = detect_ir_len(args.decoded_dir)
    noise_decoded, fs, source = decode_noise(args, ir_len)
    bands = parse_bands(args.bands, fs, MAX_HZ_EFF)
    for decoded_dir in args.decoded_dir:
        analyze_dir(decoded_dir, args, bands, fs, noise_decoded)


if __name__ == "__main__":
    main()
