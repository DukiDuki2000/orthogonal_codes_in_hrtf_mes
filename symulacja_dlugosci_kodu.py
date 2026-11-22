import os

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

import calculate_metrics as cm
from decode_bundle import bundle_mls, golay_pair, xcorr

POMIARY = ('1sesja_2zrodla', '2sesja_2zrodla', '2sesja_4zrodla')
DLUGOSC_MODELU = 16640
KODY = [('golay', r) for r in range(12, 18)] + [('mls', r) for r in (12, 14, 16, 17)]


def main():
    rows = []
    for pomiar in POMIARY:
        folder = os.path.join('wyniki', pomiar)
        run = cm.load_run_metadata(folder)
        irs = cm.load_decoded_ir(folder)
        fs = run['fs']
        starts = cm.signal_window_starts(irs, run, fs)
        noise = sf.read(os.path.join(folder, 'snr_z_ciszy', 'noise_floor_selected.wav'))[0] / (run['excitation_gain'] or 1.)
        n = round(.100 * fs)
        for sygnal, rzad in KODY:
            kody = golay_pair(rzad) if sygnal == 'golay' else [bundle_mls(rzad)]
            p_szum = sum(cm.filtered_power(noise, c) for c in kody) / len(kody) ** 2
            for speaker, h in irs['solo_golay'].items():
                h = h[:DLUGOSC_MODELU]
                odzyskana = sum(xcorr(fftconvolve(c, h), c, len(h)) / len(c) for c in kody) / len(kody)
                s = starts[speaker]
                ref = h[s:s + n]
                e_blad, e_ref = np.sum((odzyskana[s:s + n] - ref) ** 2), np.sum(ref ** 2)
                rows.append(dict(pomiar=pomiar, glosnik=speaker, sygnal=sygnal, rzad=rzad,
                                 emisja_2_zrodla_s=round(2 * sum(len(c) for c in kody) / fs, 3),
                                 SNR_solo_dB=10 * np.log10(e_ref / n / p_szum),
                                 SNR_hadamard_dB=10 * np.log10(e_ref / n / (p_szum / run['bundle_manifest']['hadamard_size'])),
                                 NRMSE_solo_procent=100 * np.sqrt((e_blad + n * p_szum) / e_ref),
                                 NRMSE_bez_szumu_procent=100 * np.sqrt(e_blad / e_ref)))
    os.makedirs(os.path.join('wyniki', 'dlugosc_kodu'), exist_ok=True)
    cm.write_csv(os.path.join('wyniki', 'dlugosc_kodu', 'symulacja.csv'), rows)


if __name__ == '__main__':
    main()
