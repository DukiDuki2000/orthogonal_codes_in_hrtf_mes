import argparse
import csv
import json
import os

import numpy as np
import samplerate
import soundfile as sf
from scipy.signal import correlate

from decode_bundle import DRIFT_QUALITY, bundle_mls, poczatki_okien, read_wav_mono
from generate_bundle import gold_family, golay_pair, reciprocal_mls

SYGNAL_MS = 100.0

ODNIESIENIE = {
    "golay_hadamard": "solo_golay",
    "mls_hadamard": "solo_mls",
    "mls_reciprocal": "solo_mls",
    "gold": "solo_mls",
}

SEGMENTY_KODU = {
    "solo_golay": ("solo_golay_spk", "golay_L"),
    "golay_hadamard": ("golay_had_row", "golay_L"),
    "solo_mls": ("solo_mls_spk", "mls_L"),
    "mls_hadamard": ("mls_had_row", "mls_L"),
    "mls_reciprocal": ("mls_reciprocal_pair", "mls_L"),
    "gold": ("gold_all", "gold_L"),
}


def load_run_metadata(decoded_dir):
    with open(os.path.join(decoded_dir, "decode_run.json"), encoding="utf-8") as f:
        return json.load(f)


def load_decoded_ir(decoded_dir):
    irs = {}
    for filename in load_run_metadata(decoded_dir)["files"]:
        method, speaker = filename[len("decoded_"):-len(".npy")].rsplit("_spk", 1)
        irs.setdefault(method, {})[int(speaker)] = np.load(os.path.join(decoded_dir, filename))
    return irs


def signal_window_starts(irs, run, fs):
    solo = [irs["solo_golay"][j] for j in sorted(irs["solo_golay"])]
    return poczatki_okien(solo, run["zero_index"], fs)


def okno_sygnalu(ir, fs, start):
    return ir[start:start + int(round(SYGNAL_MS * fs / 1000.0))]


def rmse_czasowa(ref, test, fs, start):
    r, t = okno_sygnalu(ref, fs, start), okno_sygnalu(test, fs, start)
    return float(100.0 * np.linalg.norm(t - r) / np.linalg.norm(r))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows):
    print(f"Okno analizy: {SYGNAL_MS:g} ms")
    print(f"{'Metoda':<20} {'Zrodlo':>6} {'SNR [dB]':>10} {'NRMSE [%]':>11} "
          f"{'Referencja':<12} {'Emisja [s]':>11} {'Procedura [s]':>14}")
    for row in rows:
        error = row['NRMSE_100ms_percent']
        error_text = f"{error:.2f}" if error != "" else "-"
        print(f"{row['method']:<20} {row['speaker']:>6} {row['SNR_noise_est_dB']:>10.2f} "
              f"{error_text:>11} {row['reference']:<12} "
              f"{row['emission_time_s']:>11.3f} {row['procedure_time_s']:>14.3f}")


def code_segments(meta, method):
    prefix = SEGMENTY_KODU[method][0]
    return sorted((s for s in meta["segments"] if s["name"].startswith(prefix)), key=lambda s: s["start"])


def emission_time_s(meta, method):
    return len(code_segments(meta, method)) * meta[SEGMENTY_KODU[method][1]] / meta["fs"]


def procedure_time_s(meta, method):
    sync = next(s for s in meta["segments"] if s["name"] == "sync_" + method)
    return (code_segments(meta, method)[-1]["end"] - sync["start"]) / meta["fs"]


def filtered_power(noise, code):
    noise = np.asarray(noise, dtype=np.float64)
    filtered = correlate(noise - np.mean(noise), code, mode="valid", method="fft") / len(code)
    return float(np.mean(filtered ** 2))


def noise_powers(noise, meta):
    a, b = golay_pair(meta["golay_n"])
    mls = bundle_mls(meta["mls_m"])
    golay = (filtered_power(noise, a) + filtered_power(noise, b)) / 4
    mls_power = filtered_power(noise, mls)
    k, speakers = meta["hadamard_size"], meta["speakers"]
    result = dict(solo_golay=[golay] * speakers,
                  golay_hadamard=[golay / k] * speakers,
                  solo_mls=[mls_power] * speakers,
                  mls_hadamard=[mls_power / k] * speakers,
                  gold=[filtered_power(noise, c) for c in gold_family(meta["gold_m"], speakers)])
    if any(s["name"] == "mls_reciprocal_pair" for s in meta["segments"]):
        result["mls_reciprocal"] = [mls_power, filtered_power(noise, reciprocal_mls(mls))]
    return result


def snr_estimate(observed_power, noise_power):
    if observed_power <= noise_power:
        return float("nan")
    return float(10 * np.log10((observed_power - noise_power) / noise_power))


def analyze(decoded_dir, save_csv=False, save_noise=False):
    run = load_run_metadata(decoded_dir)
    fs, meta = run["fs"], run["bundle_manifest"]
    _, y = read_wav_mono(run["recording"])
    if run["sync_mode"] == "full":
        y = samplerate.resample(y, 1 / (1 + run["drift"]["eps"]), DRIFT_QUALITY)
    offset = run["offset_samples"]
    segment = next(s for s in meta["segments"] if s["name"] == "noise_floor")
    lo = segment["start"] + round(1.0 * fs) + offset
    hi = segment["end"] - round(0.25 * fs) + offset
    noise = y[lo:hi]

    target = os.path.join(decoded_dir, "snr_z_ciszy")
    if save_csv or save_noise:
        os.makedirs(target, exist_ok=True)
    if save_noise:
        sf.write(os.path.join(target, "noise_floor_selected.wav"), noise, fs, subtype="FLOAT")
    powers = noise_powers(noise, meta)
    irs = load_decoded_ir(decoded_dir)
    starts = signal_window_starts(irs, run, fs)
    gain = run["excitation_gain"] or 1.0

    noise_rows, rows = [], []
    for method, channels in irs.items():
        for speaker, h in channels.items():
            start = starts[speaker]
            observed = float(np.mean(okno_sygnalu(h, fs, start) ** 2))
            noise_power = powers[method][speaker - 1] / gain ** 2
            snr = snr_estimate(observed, noise_power)
            reference = ODNIESIENIE.get(method)
            noise_rows.append(dict(method=method, speaker=speaker, observed_IR_power=observed,
                                   estimated_noise_IR_power=noise_power, SNR_noise_est_dB=snr))
            rows.append(dict(method=method, speaker=speaker, SNR_noise_est_dB=snr,
                             NRMSE_100ms_percent=rmse_czasowa(irs[reference][speaker], h, fs, start) if reference else "",
                             reference=reference or "-",
                             emission_time_s=emission_time_s(meta, method),
                             procedure_time_s=procedure_time_s(meta, method)))
    print_results(rows)
    if save_csv:
        write_csv(os.path.join(target, "snr_noise_floor.csv"), noise_rows)
        write_csv(os.path.join(target, "porownanie_metod.csv"), rows)
    if save_csv or save_noise:
        print(f"Zapisano: {target}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-dir", required=True)
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--save-noise", action="store_true")
    args = p.parse_args()
    analyze(args.decoded_dir, args.save_csv, args.save_noise)


if __name__ == "__main__":
    main()
