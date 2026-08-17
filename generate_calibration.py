import argparse
import json
import os
import numpy as np
import soundfile as sf


def sine_tone(fs, dur_s, freq, amp=0.8):
    n = int(round(dur_s * fs))
    t = np.arange(n, dtype=float) / fs
    sig = amp * np.sin(2.0 * np.pi * freq * t)
    fade = int(min(n // 20, fs * 0.01))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]
    return sig


def pink_noise(fs, dur_s, amp=0.5):
    n = int(round(dur_s * fs))
    white = np.random.randn(n)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    freqs[0] = 1.0
    X *= 1.0 / np.sqrt(freqs)
    sig = np.fft.irfft(X, n)
    sig = sig / np.max(np.abs(sig)) * amp
    return sig


def linear_chirp(fs, dur_s, f0=20.0, f1=20000.0, amp=0.6):
    n = int(round(dur_s * fs))
    t = np.arange(n, dtype=float) / fs
    k = (f1 - f0) / max(dur_s, 1e-12)
    sig = amp * np.sin(2.0 * np.pi * (f0 * t + 0.5 * k * t * t))
    fade = int(min(n // 10, fs * 0.005))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]
    return sig


def write_wav_multichannel(path, fs, channels, peak=None):
    data = np.column_stack([np.asarray(ch, dtype=np.float32) for ch in channels])
    if peak is not None:
        mx = float(np.max(np.abs(data))) if data.size else 1.0
        if mx > 0:
            data *= peak / mx

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    sf.write(path, data, fs, subtype='FLOAT')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speakers", type=int, default=2)
    p.add_argument("--fs", type=int, default=48000)
    p.add_argument("--out", type=str, default="calibration")
    args = p.parse_args()

    fs = args.fs
    n_spk = args.speakers
    segments = []
    channels = [[] for _ in range(n_spk)]
    cursor = [0]

    def add_segment(name, spk_data):
        max_len = max(len(d) for d in spk_data.values())
        for j in range(n_spk):
            seg = spk_data.get(j, np.zeros(max_len))
            if len(seg) < max_len:
                seg = np.pad(seg, (0, max_len - len(seg)))
            channels[j].append(seg)
        segments.append({"name": name, "start": cursor[0],
                         "end": cursor[0] + max_len, "dur_s": round(max_len / fs, 2)})
        cursor[0] += max_len

    def add_silence(name, dur_samp):
        s = np.zeros(dur_samp)
        add_segment(name, {j: s for j in range(n_spk)})


    add_silence("cisza_startowa", 3 * fs)

    for level_name, amp in [("peak_0.85", 0.85), ("peak_0.5_(-6dB)", 0.5), ("peak_0.25_(-12dB)", 0.25)]:
        for j in range(n_spk):
            tone = sine_tone(fs, 3.0, 1000.0, amp=amp)
            add_segment(f"sin1k_spk{j+1}_{level_name}", {j: tone})
            add_silence(f"cisza_po_sin1k_spk{j+1}_{level_name}", fs)

    for level_name, amp in [("peak_0.85", 0.85), ("peak_0.5_(-6dB)", 0.5)]:
        tone = sine_tone(fs, 3.0, 1000.0, amp=amp)
        add_segment(f"sin1k_ALL_{level_name}", {j: tone for j in range(n_spk)})
        add_silence(f"cisza_po_sin1k_ALL_{level_name}", fs)

    noise = pink_noise(fs, 5.0, amp=0.6)
    add_segment("pink_noise_ALL", {j: noise for j in range(n_spk)})
    add_silence("cisza_po_pink", fs)

    for j in range(n_spk):
        chirp = linear_chirp(fs, 3.0, f0=20.0, f1=20000.0, amp=0.7)
        add_segment(f"chirp_spk{j+1}", {j: chirp})
        add_silence(f"cisza_po_chirp_spk{j+1}", fs)

    add_silence("cisza_koncowa_noise_floor", 3 * fs)

    final = [np.concatenate(ch) for ch in channels]

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    wav_path = os.path.join(out_dir, "calibration.wav")
    write_wav_multichannel(wav_path, fs, final, peak=None)

    manifest_path = os.path.join(out_dir, "calibration_manifest.json")
    meta = {"speakers": n_spk, "fs": fs, "segments": segments}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
