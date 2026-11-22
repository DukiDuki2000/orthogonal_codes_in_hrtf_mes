# Generator

```
python generate_bundle.py --speakers 2 --golay-n 17 --mls-m 17 --gold-m 16 --guard 96000 --out bundle_nowy

python generate_bundle.py --speakers 4 --golay-n 17 --mls-m 17 --gold-m 16 --guard 96000 --out bundle_nowy_4
```

# Dekodowanie i metryki

## 1. sesja, 2 źródła

```
python decode_bundle.py --bundle-dir bundle_out_1_sesja_2_glosniki --recording nagrania/1sesja.wav --ir-len 96000 --all-methods --plot --out-dir wyniki/1sesja_2zrodla

python calculate_metrics.py --decoded-dir wyniki/1sesja_2zrodla --save-csv --save-noise
```

## 2. sesja, 2 źródła

```
python decode_bundle.py --bundle-dir bundle_out_2_sesja_2_glosniki --recording nagrania/2sesja_2zrodla.wav --ir-len 96000 --all-methods --plot --out-dir wyniki/2sesja_2zrodla

python calculate_metrics.py --decoded-dir wyniki/2sesja_2zrodla --save-csv --save-noise
```

## 2. sesja, 4 źródła

```
python decode_bundle.py --bundle-dir bundle_out_2_sesja_4_glosniki --recording nagrania/2sesja4spk.wav --ir-len 96000 --all-methods --plot --out-dir wyniki/2sesja_4zrodla

python calculate_metrics.py --decoded-dir wyniki/2sesja_4zrodla --save-csv --save-noise
```

# Analizy dodatkowe

Wymagają wyników dekodowania i metryk ze wszystkich trzech pomiarów.

```
python analiza_golda.py

python symulacja_dlugosci_kodu.py
```

# Symulacja HRTF

```
python generate_bundle.py --speakers 2 --golay-n 17 --mls-m 17 --gold-m 16 --guard 96000 --out wyniki/symulacja_hrtf_aktualna/bundle

python simulate_recording.py --bundle-dir wyniki/symulacja_hrtf_aktualna/bundle --out wyniki/symulacja_hrtf_aktualna/lewe.wav --sofa P0001_FreeFieldComp_48kHz.sofa --ear left --azimuths 0 90 --noise-db 60

python simulate_recording.py --bundle-dir wyniki/symulacja_hrtf_aktualna/bundle --out wyniki/symulacja_hrtf_aktualna/prawe.wav --sofa P0001_FreeFieldComp_48kHz.sofa --ear right --azimuths 0 90 --noise-db 60

python decode_bundle.py --bundle-dir wyniki/symulacja_hrtf_aktualna/bundle --recording wyniki/symulacja_hrtf_aktualna/lewe.wav --plot --out-dir wyniki/symulacja_hrtf_aktualna/lewe

python decode_bundle.py --bundle-dir wyniki/symulacja_hrtf_aktualna/bundle --recording wyniki/symulacja_hrtf_aktualna/prawe.wav --plot --out-dir wyniki/symulacja_hrtf_aktualna/prawe

python calculate_metrics.py --decoded-dir wyniki/symulacja_hrtf_aktualna/lewe --save-csv

python calculate_metrics.py --decoded-dir wyniki/symulacja_hrtf_aktualna/prawe --save-csv

python raport_symulacji_hrtf.py
```