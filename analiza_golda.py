import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

import calculate_metrics as cm
from decode_bundle import bundle_mls, compute_fft_mag_db, xcorr
from generate_bundle import gold_family, reciprocal_mls

POMIARY = ('1sesja_2zrodla', '2sesja_2zrodla', '2sesja_4zrodla')
OKNA_MS = (50, 100, 200, 500, 1000, 2000)
OKNO = 4800


def wczytaj(pomiar):
    folder = os.path.join('wyniki', pomiar)
    run = cm.load_run_metadata(folder)
    irs = cm.load_decoded_ir(folder)
    return run, irs, cm.signal_window_starts(irs, run, run['fs'])


def nrmse(test, ref):
    return 100 * np.linalg.norm(test - ref) / np.linalg.norm(ref)


def okna_golda(pomiar):
    run, irs, starts = wczytaj(pomiar)
    fs = run['fs']
    szum = {(r['method'], int(r['speaker'])): float(r['estimated_noise_IR_power'])
            for r in csv.DictReader(open(f'wyniki/{pomiar}/snr_z_ciszy/snr_noise_floor.csv', encoding='utf-8-sig'), delimiter=';')}
    target = os.path.join('wyniki', 'gold', 'okna', pomiar)
    os.makedirs(target, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for speaker, ax in zip((1, 2), axes):
        s = starts[speaker]
        gold, mls = irs['gold'][speaker], irs['solo_mls'][speaker]
        for ms in OKNA_MS:
            n = round(ms * fs / 1000)
            rows.append(dict(zrodlo=speaker, okno_ms=ms,
                             NRMSE_gold_procent=nrmse(gold[s:s + n], mls[s:s + n]),
                             SNR_gold_dB=cm.snr_estimate(np.mean(gold[s:s + n] ** 2), szum['gold', speaker]),
                             SNR_solo_mls_dB=cm.snr_estimate(np.mean(mls[s:s + n] ** 2), szum['solo_mls', speaker])))
        for h, label, color in ((mls, 'Solo MLS', 'black'), (gold, 'Gold', 'tab:orange')):
            f, level = compute_fft_mag_db(h[s:s + round(.1 * fs)], fs, 20000)
            ax.plot(f, level, label=label, color=color, linewidth=.7, alpha=.85)
        ax.set_xlim(60 if pomiar.startswith('1sesja') else 1500, 20000)
        ax.set_title(f'Źródło {speaker}')
        ax.set_ylabel('Moduł odpowiedzi [dB]')
        ax.grid(alpha=.3)
        ax.legend()
    axes[-1].set_xlabel('Częstotliwość [Hz]')
    fig.suptitle('Gold i Solo MLS — okno analizy: 100 ms')
    fig.tight_layout()
    fig.savefig(os.path.join(target, 'gold_vs_solo_mls_100ms.png'), dpi=160)
    plt.close(fig)
    cm.write_csv(os.path.join(target, 'okna.csv'), rows)


def dekoduj(y, kod, n):
    return xcorr(y, kod, n) / len(kod)


def model_liniowy():
    mls = bundle_mls(17)
    rows = []
    for pomiar in POMIARY:
        _, irs, starts = wczytaj(pomiar)
        okno = lambda x, j: x[starts[j]:starts[j] + OKNO]
        h = irs['solo_golay']
        n = len(h[1])
        kody = {'gold': dict(enumerate(gold_family(16, len(h)), 1))}
        if 'mls_reciprocal' in irs:
            kody['mls_reciprocal'] = {1: mls, 2: reciprocal_mls(mls)}
        for metoda, kod in kody.items():
            for j in kod:
                ref = okno(dekoduj(fftconvolve(h[j], mls), mls, n), j)
                wlasne = okno(dekoduj(fftconvolve(h[j], kod[j]), kod[j], n), j)
                przesluch = okno(dekoduj(sum(fftconvolve(h[i], kod[i]) for i in kod if i != j), kod[j], n), j)
                blad = wlasne + przesluch - ref
                rows.append({
                    'pomiar': pomiar, 'metoda': metoda, 'glosnik': j,
                    'NRMSE_zmierzone_%': round(nrmse(okno(irs[metoda][j], j), okno(irs['solo_mls'][j], j)), 2),
                    'NRMSE_symulacja_%': round(100 * np.linalg.norm(blad) / np.linalg.norm(ref), 2),
                    'NRMSE_badane_zrodlo_%': round(nrmse(wlasne, ref), 2),
                    'przesluch_%': round(100 * np.linalg.norm(przesluch) / np.linalg.norm(ref), 2),
                    'udzial_przesluchu_%': round(100 * np.sum(przesluch ** 2) / np.sum(blad ** 2), 1),
                })
    return rows


def main():
    for pomiar in POMIARY[:2]:
        okna_golda(pomiar)
    cm.write_csv(os.path.join('wyniki', 'gold', 'model_liniowy_i_przesluch.csv'), model_liniowy())


if __name__ == '__main__':
    main()
