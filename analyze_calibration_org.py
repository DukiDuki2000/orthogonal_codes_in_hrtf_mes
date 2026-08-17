import argparse
import json
import os
import numpy as np
import soundfile as sf


def read_wav_mono(path):
    data, fs = sf.read(path, always_2d=False)
    data = np.asarray(data)

    if data.ndim > 1:
        data = data[:, 0]

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float64) / max(1, abs(info.min), info.max)

    return data.astype(np.float64), fs


def dbfs(signal):
    peak = np.max(np.abs(signal))
    if peak < 1e-12:
        return -120.0
    return 20.0 * np.log10(peak)


def rms_dbfs(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-12:
        return -120.0
    return 20.0 * np.log10(rms)


def count_clips(signal, threshold=0.99):
    return int(np.sum(np.abs(signal) >= threshold))


def thd_percent(signal, fs, fundamental_hz, n_harmonics=5):
    N = len(signal)
    if N < 1024:
        return None
    window = np.hanning(N)
    X = np.abs(np.fft.rfft(signal * window)) / N
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    def find_peak_power(target_hz, tolerance=20):
        mask = (freqs >= target_hz - tolerance) & (freqs <= target_hz + tolerance)
        if not np.any(mask):
            return 0.0
        return float(np.max(X[mask]) ** 2)

    p_fund = find_peak_power(fundamental_hz)
    if p_fund < 1e-20:
        return None

    p_harm = 0.0
    for h in range(2, n_harmonics + 2):
        p_harm += find_peak_power(fundamental_hz * h)

    return 100.0 * np.sqrt(p_harm / p_fund)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recording", required=True, help="Ścieżka do nagranego WAV")
    p.add_argument("--manifest", required=True, help="Ścieżka do calibration_manifest.json")
    p.add_argument("--out", default="./calibration_report", help="Katalog wyjściowy")
    args = p.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rec, rec_fs = read_wav_mono(args.recording)
    man_fs = manifest["fs"]
    segments = manifest["segments"]

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)


    if rec_fs != man_fs:
        print(f"Zły sample rate")
        return 0

    results = []

    for seg in segments:
        name = seg["name"]
        start = seg["start"]
        end = seg["end"]
        chunk = rec[start:end]
        peak = dbfs(chunk)
        rms = rms_dbfs(chunk)
        clips = count_clips(chunk)

        thd = None
        if "sin1k" in name and "cisza" not in name:
            thd = thd_percent(chunk, rec_fs, 1000.0)

        results.append({
            "name": name,
            "peak_dbfs": round(peak, 2),
            "rms_dbfs": round(rms, 2),
            "clips": clips,
            "thd_percent": round(thd, 3) if thd is not None else None,
        })

    print(f"Nagranie: {args.recording}")
    print(f"Sample rate: {rec_fs} Hz")
    print(f"Długość: {len(rec) / rec_fs:.2f} s ({len(rec)} próbek)")

    for r in results:
        print(
            f"{r['name']:25} | peak={r['peak_dbfs']:>6.2f} dBFS | "
            f"rms={r['rms_dbfs']:>6.2f} dBFS | clips={r['clips']} | "
            f"THD={r['thd_percent'] if r['thd_percent'] is not None else 'n/a'}"
        )

    noise_segs = [s for s in segments if "cisza_koncowa" in s["name"]]
    if noise_segs:
        ns = noise_segs[0]
        if ns["end"] <= len(rec):
            noise_chunk = rec[ns["start"]:ns["end"]]
            nf_rms = rms_dbfs(noise_chunk)
            print(f"\nNoise floor:   {nf_rms:.1f} dBFS (RMS)")
            sin_segs = [r for r in results if "sin1k" in r["name"] and "ALL" not in r["name"]
                        and r["rms_dbfs"] > -60]
            if sin_segs:
                best_rms = max(r["rms_dbfs"] for r in sin_segs)
                snr = best_rms - nf_rms
                print(f"SNR:           {snr:.1f} dB")

    total_clips = sum(r["clips"] for r in results)
    print(f"\nClipping: {'Tak' if total_clips > 0 else 'Nie'}")

    sin_results = [r for r in results if "sin1k" in r["name"] and "peak_0.85" in r["name"]
                   and "cisza" not in r["name"] and "ALL" not in r["name"]]
    if sin_results:
        avg_peak = np.mean([r["peak_dbfs"] for r in sin_results])
        print(f"\nŚredni peak (sinus 0dB, solo): {avg_peak:.1f} dBFS")
        if avg_peak > -3:
            print("  -> 5")
        elif avg_peak > -6:
            print("  -> 4")
        elif avg_peak > -12:
            print("  -> 3")
        elif avg_peak > -20:
            print("  -> 2")
        else:
            print("  -> 1")


if __name__ == "__main__":
    main()
