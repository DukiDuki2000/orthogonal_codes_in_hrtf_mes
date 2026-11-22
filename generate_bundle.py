import argparse
import json
import os
import numpy as np
import soundfile as sf


def golay_pair(n: int):
    A, B = np.array([1.0]), np.array([1.0])
    for _ in range(n):
        A, B = np.concatenate([A, B]), np.concatenate([A, -B])
    return A, B


MLS_TAPS = {
    8: [8, 6, 5, 4], 9: [9, 5], 10: [10, 7], 12: [12, 6, 4, 1],
    14: [14, 5, 3, 1], 16: [16, 14, 13, 11], 17: [17, 14], 18: [18, 11],
}

GOLD_PAIRS = {
    5: ([5, 3], [5, 4, 3, 2]),
    6: ([6, 5], [6, 5, 2, 1]),
    7: ([7, 3], [7, 3, 2, 1]),
    8: ([8, 6, 5, 4], [8, 7, 6, 1]),
    9: ([9, 4], [9, 6, 4, 3]),
    10: ([10, 7], [10, 9, 8, 6]),
    12: ([12, 6, 4, 1], [12, 11, 10, 4]),
    16: ([16, 14, 13, 11], [16, 15, 13, 4]),
}


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


def reciprocal_mls(seq):
    if len(seq) <= 1:
        return np.array(seq, copy=True)
    return np.concatenate([seq[:1], seq[:0:-1]])


def gold_family(m, n_codes):
    t1, t2 = GOLD_PAIRS[m]
    mls1 = generate_mls(t1, m)
    mls2 = generate_mls(t2, m)
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


def next_pow2(n):
    return int(np.power(2, np.ceil(np.log2(max(1, n)))))


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


def linear_chirp(fs, dur_s, f0=600.0, f1=6000.0, amp=0.6):
    n = int(round(dur_s * fs))
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n, dtype=float) / fs
    k = (f1 - f0) / max(dur_s, 1e-12)
    sig = amp * np.sin(2.0 * np.pi * (f0 * t + 0.5 * k * t * t))
    fade = int(min(n // 10, fs * 0.005))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]
    return sig


def write_wav_multichannel(path, fs, channels, peak=0.85):
    data = np.column_stack([np.asarray(ch, dtype=np.float32) for ch in channels])

    mx = float(np.max(np.abs(data))) if data.size else 1.0
    if mx > 0:
        data *= peak / mx

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    sf.write(path, data, fs, subtype='FLOAT')
    return float(peak / mx) if mx > 0 else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speakers", type=int, default=2, help="Liczba głośników")
    p.add_argument("--fs", type=int, default=48000)
    p.add_argument("--golay-n", type=int, default=16, help="Golay: 2^n próbek")
    p.add_argument("--mls-m", type=int, default=14, help="MLS: 2^m-1 próbek")
    p.add_argument("--gold-m", type=int, default=10, help="Gold: 2^m-1 próbek")
    p.add_argument("--guard", type=int, default=4096, help="Guard [próbki]")
    p.add_argument("--sync-ms", type=float, default=40.0, help="Chirp sync [ms]")
    p.add_argument("--gap-ms", type=float, default=200.0, help="Cisza między blokami [ms]")
    p.add_argument("--start-delay-ms", type=float, default=0.0, help="Cisza na początku pliku [ms]")
    p.add_argument("--noise-floor-ms", type=float, default=5000.0, help="Pomiar szumu tła (cisza) [ms]")
    p.add_argument("--thd-freqs", type=str, default="100,250,500,1000,2000,4000,8000",
                   help="Częstotliwości sinusów")
    p.add_argument("--thd-dur-ms", type=float, default=2000.0, help="Czas trwania sinusa THD [ms]")
    p.add_argument("--peak", type=float, default=0.85)
    p.add_argument("--out", type=str, default="bundle_out")
    args = p.parse_args()

    n_spk = args.speakers

    fs = args.fs
    guard = np.zeros(args.guard)
    gap = np.zeros(int(round(args.gap_ms / 1000 * fs)))
    sync = linear_chirp(fs, args.sync_ms / 1000)

    A, B = golay_pair(args.golay_n)
    L_golay = len(A)

    if args.mls_m not in MLS_TAPS:
        raise ValueError(f"--mls-m={args.mls_m} niedostępne.")
    mls = generate_mls(MLS_TAPS[args.mls_m], args.mls_m)
    mls_reciprocal = reciprocal_mls(mls)
    L_mls = len(mls)

    gold_codes = gold_family(args.gold_m, n_spk)
    L_gold = len(gold_codes[0])

    h_size = next_pow2(n_spk)
    H = hadamard_matrix(h_size)

    channels = [[] for _ in range(n_spk)]
    manifest = []
    cursor = [0]

    def append_all(name, kind, spk_data):
        if spk_data is None:
            return
        max_len = max(len(d) for d in spk_data.values()) if spk_data else 0
        if max_len == 0:
            return
        for j in range(n_spk):
            if j in spk_data:
                seg = np.asarray(spk_data[j], dtype=float)
                if len(seg) < max_len:
                    seg = np.pad(seg, (0, max_len - len(seg)))
            else:
                seg = np.zeros(max_len)
            channels[j].append(seg)
        manifest.append({"name": name, "kind": kind,
                         "start": cursor[0], "end": cursor[0] + max_len, "fs": fs})
        cursor[0] += max_len

    def append_silence(name, length):
        s = np.zeros(length)
        append_all(name, "silence", {j: s for j in range(n_spk)})

    def append_sync(label):
        append_all(f"sync_{label}", "sync", {j: sync for j in range(n_spk)})
        append_silence(f"gap_after_sync_{label}", len(gap))


    start_delay_samp = int(round(args.start_delay_ms / 1000 * fs))
    if start_delay_samp > 0:
        append_silence("start_delay", start_delay_samp)
    noise_floor_samp = int(round(args.noise_floor_ms / 1000 * fs))
    if noise_floor_samp > 0:
        append_sync("noise_floor")
        append_silence("noise_floor", noise_floor_samp)
        append_silence("gap_after_noise_floor", len(gap))


    append_sync("solo_golay")
    for j in range(n_spk):
        seg_a = np.concatenate([A, guard])
        append_all(f"solo_golay_spk{j+1}_A", "solo_golay_A",
                   {j: seg_a})
        seg_b = np.concatenate([B, guard])
        append_all(f"solo_golay_spk{j+1}_B", "solo_golay_B",
                   {j: seg_b})
    append_silence("gap_after_solo_golay", len(gap))


    append_sync("solo_mls")
    for j in range(n_spk):
        seg = np.concatenate([mls, guard])
        append_all(f"solo_mls_spk{j+1}", "solo_mls",
                   {j: seg})
    append_silence("gap_after_solo_mls", len(gap))


    append_sync("golay_hadamard")
    for i in range(h_size):
        data_a = {}
        for j in range(n_spk):
            data_a[j] = np.concatenate([H[i, j] * A, guard])
        append_all(f"golay_had_row{i+1}_A", "golay_hadamard_A", data_a)
        data_b = {}
        for j in range(n_spk):
            data_b[j] = np.concatenate([H[i, j] * B, guard])
        append_all(f"golay_had_row{i+1}_B", "golay_hadamard_B", data_b)
    append_silence("gap_after_golay_had", len(gap))


    append_sync("mls_hadamard")
    for i in range(h_size):
        data = {}
        for j in range(n_spk):
            data[j] = np.concatenate([H[i, j] * mls, guard])
        append_all(f"mls_had_row{i+1}", "mls_hadamard", data)
    append_silence("gap_after_mls_had", len(gap))


    if n_spk == 2:
        append_sync("mls_reciprocal")
        data = {
            0: np.concatenate([mls, guard]),
            1: np.concatenate([mls_reciprocal, guard]),
        }
        append_all("mls_reciprocal_pair", "mls_reciprocal", data)
        append_silence("gap_after_mls_reciprocal", len(gap))


    append_sync("gold")
    data = {}
    for j in range(n_spk):
        data[j] = np.concatenate([gold_codes[j], guard])
    append_all("gold_all", "gold", data)
    append_silence("gap_after_gold", len(gap))


    thd_freqs = [int(f) for f in args.thd_freqs.split(",")]
    thd_dur = args.thd_dur_ms / 1000.0
    if thd_dur > 0 and thd_freqs:
        append_sync("thd")
        for j in range(n_spk):
            for freq in thd_freqs:
                tone = sine_tone(fs, thd_dur, freq)
                seg = np.concatenate([tone, guard])
                append_all(f"thd_spk{j+1}_{freq}Hz", "thd", {j: seg})
        append_silence("gap_after_thd", len(gap))


    final = [np.concatenate(ch) for ch in channels]
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "bundle.wav")
    excitation_gain = write_wav_multichannel(wav_path, fs, final, peak=args.peak)
    manifest_path = os.path.join(out_dir, "manifest.json")
    meta = {
        "excitation_gain": excitation_gain,
        "speakers": n_spk,
        "fs": fs,
        "golay_n": args.golay_n,
        "golay_L": L_golay,
        "mls_m": args.mls_m,
        "mls_L": L_mls,
        "mls_reciprocal_L": L_mls,
        "gold_m": args.gold_m,
        "gold_L": L_gold,
        "hadamard_size": h_size,
        "guard": args.guard,
        "segments": manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
