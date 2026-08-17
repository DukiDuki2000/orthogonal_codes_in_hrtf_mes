# Generator

```
python generate_bundle.py --speakers 2 --golay-n 17 --mls-m 17 --gold-m 16 --guard 96000 --out bundle_nowy

python generate_bundle.py --speakers 4 --golay-n 17 --mls-m 17 --gold-m 16 --guard 96000 --out bundle_nowy_2
```

## 1. sesja, 2 źródła

```
python decode_bundle.py --bundle-dir bundle_out_1_sesja_2_glosniki --recording nagrania/1sesja.wav --all-methods --out-dir wyniki/1sesja --fmin-plot 60 --plot

python calculate_metrics.py --decoded-dir wyniki/1sesja --bundle-manifest bundle_out_1_sesja_2_glosniki/manifest.json --calibration-recording calibration/1_sesja_cal.wav --calibration-manifest calibration/calibration_manifest.json
```

## 2. sesja, 2 źródła

```
python decode_bundle.py --bundle-dir bundle_out_2_sesja_2_glosniki --recording nagrania/2sesja2spk.wav --ir-len 24000 --all-methods --out-dir wyniki/2sesja/2spk

python calculate_metrics.py --decoded-dir wyniki/2sesja/2spk --bundle-manifest bundle_out_2_sesja_2_glosniki/manifest.json --noise-wav nagrania/2sesja2spk.wav --bands "2000,4000,8000,16000"
```

## 2. sesja, 4 źródła

```
python decode_bundle.py --bundle-dir bundle_out_2_sesja_4_glosniki --recording nagrania/2sesja4spk.wav --ir-len 24000 --all-methods --out-dir wyniki/2sesja/4spk

python calculate_metrics.py --decoded-dir wyniki/2sesja/4spk --bundle-manifest bundle_out_2_sesja_4_glosniki/manifest.json --noise-wav calibration/2_sesja_cal.wav --bands "2000,4000,8000,16000"
```


# Sprawdzenie toru pomiarowego

```
python generate_calibration.py --speakers 2 --fs 48000 --out calibration/1sesja_cal.wav

python analyze_calibration_org.py --recording calibration/1_sesja_cal.wav --manifest calibration/calibration_manifest.json --out wyniki/kalibracja
```

# Symulacja

```
python simulate_recording.py --bundle-dir bundle_nowy --out nagrania/symulacja.wav --ir-len 4096 --noise-db 60

python simulate_recording.py --bundle-dir bundle_nowy --out nagrania/symulacja.wav --sofa P0001_FreeFieldComp_48kHz.sofa
```