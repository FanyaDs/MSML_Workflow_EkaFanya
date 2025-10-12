# MSML Workflow CI — Eka Fanya

Otomatisasi retraining model Machine Learning setiap kali ada perubahan di branch `main`.

## Struktur
- `.github/workflows/ml_ci.yml` — GitHub Actions untuk retraining otomatis.
- `MLProject/` — MLflow Project dan environment.
- `modelling.py` — Script training model Random Forest.
- `namadataset_preprocessing/` — Data hasil preprocessing dari Eksperimen.

## Langkah Kerja
1. Commit & push perubahan ke branch `main`.
2. Workflow otomatis jalan via GitHub Actions.
3. Model dilatih ulang menggunakan MLflow Project.
4. Artefak hasil training tersimpan otomatis di Actions.
