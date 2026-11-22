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


def write_wav_mono(path, fs, x):
    x = np.asarray(x, dtype=float).ravel()
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
    p.add_argument("--ear", choices=["left", "right"], default="left")
    p.add_argument("--azimuths", type=float, nargs="+", help="Azymuty źródeł [stopnie]")
    p.add_argument("--elevation", type=float, default=0.0)
    args = p.parse_args()

    bundle_dir = os.path.abspath(args.bundle_dir)
    wav_path = os.path.join(bundle_dir, "bundle.wav")
    manifest_path = os.path.join(bundle_dir, "manifest.json")

    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_spk = meta["speakers"]
    fs, data = read_wav_multichannel(wav_path)
    n_samp, n_ch = data.shape
    if n_ch < n_spk or fs != meta["fs"]:
        p.error("Niezgodne kanały lub częstotliwość WAV")
    ir_len = args.ir_len
    irs = []
    selected = []
    if args.sofa:
        with h5py.File(args.sofa, "r") as sofa_file:
            IR = np.array(sofa_file["Data.IR"])
            positions = np.array(sofa_file["SourcePosition"])
            sofa_fs = np.array(sofa_file["Data.SamplingRate"])
            delays = np.array(sofa_file["Data.Delay"])
            position_type = sofa_file["SourcePosition"].attrs.get("Type", "")
            position_units = sofa_file["SourcePosition"].attrs.get("Units", "")
        if not np.all(sofa_fs == fs) or np.any(delays != 0):
            p.error("SOFA wymaga zgodnego fs i zerowego Data.Delay")
        if isinstance(position_type, bytes):
            position_type = position_type.decode()
        if isinstance(position_units, bytes):
            position_units = position_units.decode()
        if position_type != "spherical" or position_units != "degree, degree, metre":
            p.error("Wymagane pozycje sferyczne w stopniach i metrach")
        azimuths = args.azimuths if args.azimuths is not None else np.linspace(0, 360, n_spk, endpoint=False)
        if len(azimuths) != n_spk:
            p.error("Podaj jeden azymut dla każdego źródła")
        az, el = np.deg2rad(positions[:, 0]), np.deg2rad(positions[:, 1])
        target_el = np.deg2rad(args.elevation)
        idx = [int(np.argmax(np.sin(el)*np.sin(target_el) + np.cos(el)*np.cos(target_el)*np.cos(az-np.deg2rad(a)))) for a in azimuths]
        if len(set(idx)) != n_spk:
            p.error("Wybrane kierunki wskazują tę samą HRIR")
        for j, i in enumerate(idx):
            irs.append(IR[i, 0 if args.ear == "left" else 1, :].astype(float))
            selected.append(dict(index=i, requested_azimuth=float(azimuths[j]), position=positions[i].tolist()))
            print(f"  IR spk{j+1}: SOFA pozycja {i}, len={irs[-1].size}")
    else:
        for j in range(n_spk):
            delay = 2.0 + j * 1.5
            decay = 80.0 + j * 20.0
            irs.append(make_ir(fs, ir_len, delay, decay, seed=args.seed + j + 1))

    out_path = os.path.abspath(args.out)
    ir_dir = os.path.splitext(out_path)[0] + "_true_irs"
    os.makedirs(ir_dir, exist_ok=True)
    for j, ir in enumerate(irs):
        np.save(os.path.join(ir_dir, f"true_h{j+1}.npy"), ir)

    y = np.zeros(n_samp + max(map(len, irs)) - 1)
    for j in range(n_spk):
        conv = fftconvolve(data[:, j], irs[j])
        y[:len(conv)] += conv

    rng = np.random.default_rng(args.seed + 999)
    clean_power = float(np.mean(y**2))
    y = add_awgn(y, args.noise_db, rng)
    write_wav_mono(out_path, fs, y)
    info = dict(fs=fs, sofa=os.path.abspath(args.sofa) if args.sofa else None,
                ear=args.ear, selected=selected, seed=args.seed, noise_db=args.noise_db,
                clean_recording_power=clean_power, expected_noise_power=clean_power/10**(args.noise_db/10) if args.noise_db < 200 else 0,
                output_gain=1.0, true_irs=ir_dir, ir_lengths=list(map(len, irs)),
                clock_drift_ppm=0.0, recording_offset_samples=0)
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Zapisano: {out_path} ({len(y)} próbek, {len(y)/fs:.2f}s)")


if __name__ == "__main__":
    main()
