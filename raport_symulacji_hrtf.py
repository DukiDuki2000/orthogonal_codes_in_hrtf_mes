import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import calculate_metrics as cm

KATALOG = os.path.join('wyniki', 'symulacja_hrtf_aktualna')
METODY = {'solo_golay': 'Solo Golay', 'golay_hadamard': 'Golay + Hadamard'}


def main():
    wiersze = []
    for ucho in ('lewe', 'prawe'):
        folder = os.path.join(KATALOG, ucho)
        run = cm.load_run_metadata(folder)
        irs = cm.load_decoded_ir(folder)
        fs = run['fs']
        starts = cm.signal_window_starts(irs, run, fs)
        metryki = {(r['method'], int(r['speaker'])): r for r in csv.DictReader(
            open(os.path.join(folder, 'snr_z_ciszy', 'porownanie_metod.csv'), encoding='utf-8-sig'), delimiter=';')}
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        for j, ax in enumerate(axes, 1):
            hrir = np.load(os.path.join(KATALOG, f'{ucho}_true_irs', f'true_h{j}.npy'))
            reference = np.zeros_like(irs['solo_golay'][j])
            reference[run['zero_index']:run['zero_index'] + len(hrir)] = hrir
            s = starts[j]
            frequency = np.fft.rfftfreq(len(cm.okno_sygnalu(reference, fs, s)), 1 / fs)
            for h, label, style in [(reference, 'HRIR z SOFA', '-'), (irs['solo_golay'][j], 'Solo Golay', '--'),
                                    (irs['golay_hadamard'][j], 'Golay + Hadamard', ':')]:
                mag = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(cm.okno_sygnalu(h, fs, s))), 1e-15))
                ax.semilogx(frequency[1:], mag[1:], style, label=label, linewidth=1.3)
            for metoda, label in METODY.items():
                r = metryki[metoda, j]
                nrmse_solo = f"{float(r['NRMSE_100ms_percent']):.4f}" if r['NRMSE_100ms_percent'] else '-'
                wiersze.append(f"{ucho:<6} {(j - 1) * 90:>6} {label:<17} {float(r['SNR_noise_est_dB']):>9.2f} "
                               f"{cm.rmse_czasowa(reference, irs[metoda][j], fs, s):>15.4f} {nrmse_solo:>15}")
            ax.set(xlim=(100, 20000), ylabel='Moduł [dB]', title=f'Ucho {ucho}, azymut {(j - 1) * 90}°, elewacja 0°')
            ax.grid(True, which='both', alpha=.25)
            ax.legend()
        axes[-1].set_xlabel('Częstotliwość [Hz]')
        fig.savefig(os.path.join(KATALOG, f'{ucho}_ucho.png'), dpi=180)
        plt.close(fig)
    print(f"{'Ucho':<6} {'Azymut':>6} {'Metoda':<17} {'SNR [dB]':>9} {'NRMSE SOFA [%]':>15} {'NRMSE Solo [%]':>15}")
    print('\n'.join(wiersze))


if __name__ == '__main__':
    main()
