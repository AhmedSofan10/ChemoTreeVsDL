# PrimeNet integration (vendored)

PrimeNet ([Roy Chowdhury et al., AAAI 2023](https://github.com/ranakroychowdhury/PrimeNet)) is integrated for **MIMIC-IV neutropenic fever** using the same cohort, folds, and top-100 labs as STraTS.

## Layout

- `third_party/PrimeNet/` — vendored upstream TimeBERT (pinned commit in `third_party/PrimeNet/VENDOR.md`)
- `ts_model_training/primenet/` — export, train loop, metrics (ChemoTree data paths)
- `colab_primenet_train.py` — Google Colab entry point
- `notebooks/primenet_mimic_iv_colab.ipynb` — Colab notebook
- `scripts/prepare_mimic_from_raw.py` — build `MIMIC_IV/saved_data/` from `data/raw/`

## Train (local)

```bash
python -m ts_model_training.main \
  --dataset MIMIC_IV \
  --cohort mimic_cohort_NF_30_days \
  --fold 0 \
  --model_type primenet \
  --feature_threshold \
  --grid none \
  --prefix primenet_run
```

## Colab

```bash
pip install -r requirements-colab.txt
python colab_primenet_train.py --fold 0 --fast          # smoke test
python colab_primenet_train.py --fold 0 --prefix run  # full fold
```

Upload raw CSVs to `data/raw/` or copy `MIMIC_IV/saved_data/` from Drive, then run the script.

## Scope (v1)

- `grid=none` only (nested grid not supported for PrimeNet yet)
- MIMIC-IV NF cohort first
- Fair comparison: PrimeNet vs STraTS on **same folds**, not vs paper MIMIC-III mortality AUROC
