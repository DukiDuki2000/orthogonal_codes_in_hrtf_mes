import argparse
import json
import os
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import h5py

def read_wav_multichannel(path):
    data, fs = sf.read(path, always_2d=True)
    return fs, np.asarray(data, dtype=np.float64)


def write_wav_mono(path, fs, x, peak=0.95):
    x = np.asarray(x, dtype=float).ravel()
    mx = float(np.max(np.abs(x))) if x.size else 1.0
    if mx > 0:
        x = x * (peak / mx)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, x.astype(np.float32), fs, subtype='FLOAT')

def make_ir(fs, ir_len, delay_ms, decay_ms, seed):
    rng = np.random.default_rng(seed)
    ir = np.zeros(ir_len)
    d = int(delay_ms / 1000.0 * fs)
    if 0 <= d < ir_len:
        ir[d] = 1.0
    tau = max(decay_ms / 1000.0 * fs, 1.0)
    ir += rng.standard_normal(ir_len) * 0.15 * np.exp(-np.arange(ir_len) / tau)
    return ir


def add_awgn(signal, noise_db, rng):
    if noise_db >= 200:
        return signal.copy()
    sig_p = max(float(np.mean(signal**2)), 1e-30)
    noise_p = sig_p / (10 ** (noise_db / 10))
    return signal + rng.standard_normal(len(signal)) * np.sqrt(noise_p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", required=True, help="Folder z bundle.wav i manifest.json")
    p.add_argument("--out", default="recording.wav", help="Ścieżka wyjściowego mono WAV")
    p.add_argument("--ir-len", type=int, default=4096, help="Długość IR")
    p.add_argument("--noise-db", type=float, default=60.0, help="SNR [dB]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sofa", default=None,
                   help="Plik SOFA")
    args = p.parse_args()

    bundle_dir = os.path.abspath(args.bundle_dir)
    wav_path = os.path.join(bundle_dir, "bundle.wav")
    manifest_path = os.path.join(bundle_dir, "manifest.json")

    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_spk = meta["speakers"]
    fs, data = read_wav_multichannel(wav_path)
    n_samp, n_ch = data.shape
    ir_len = args.ir_len
    irs = []
    if args.sofa:
        with h5py.File(args.sofa, "r") as sofa_file:
            IR = np.array(sofa_file["Data.IR"])
        M = IR.shape[0]
        idx = np.linspace(0, M - 1, n_spk).astype(int)
        for j, i in enumerate(idx):
            irs.append(IR[i, 0, :].astype(float))   # ucho 0
            print(f"  IR spk{j+1}: SOFA pozycja {i}, len={irs[-1].size}")
    else:
        for j in range(n_spk):
            delay = 2.0 + j * 1.5
            decay = 80.0 + j * 20.0
            irs.append(make_ir(fs, ir_len, delay, decay, seed=args.seed + j + 1))

    ir_dir = os.path.join(bundle_dir, "true_irs")
    os.makedirs(ir_dir, exist_ok=True)
    for j, ir in enumerate(irs):
        np.save(os.path.join(ir_dir, f"true_h{j+1}.npy"), ir)

    y = np.zeros(n_samp + ir_len - 1)
    for j in range(n_spk):
        conv = fftconvolve(data[:, j], irs[j])
        y[:len(conv)] += conv

    rng = np.random.default_rng(args.seed + 999)
    y = add_awgn(y, args.noise_db, rng)
    out_path = os.path.abspath(args.out)
    write_wav_mono(out_path, fs, y)
    print(f"Zapisano: {out_path} ({len(y)} próbek, {len(y)/fs:.2f}s)")


if __name__ == "__main__":
    main()
