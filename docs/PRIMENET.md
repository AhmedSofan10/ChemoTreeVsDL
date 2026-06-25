# PrimeNet integration (vendored)

PrimeNet ([Roy Chowdhury et al., AAAI 2023](https://github.com/ranakroychowdhury/PrimeNet)) is integrated for **MIMIC-IV neutropenic fever** using the same cohort, folds, and top-100 labs as STraTS.

## Layout

- `third_party/PrimeNet/` — vendored upstream TimeBERT (pinned commit in `third_party/PrimeNet/VENDOR.md`)
- `ts_model_training/primenet/` — export, train loop, metrics (ChemoTree data paths)
- `colab_primenet_train.py` — Google Colab entry point
- `notebooks/primenet_mimic_iv_colab.ipynb` — Colab notebook
- `scripts/prepare_mimic_from_raw.py` — build `MIMIC_IV/saved_data/` from `data/raw/`

## Train (local GPU)

### 1. One-time setup

```bash
cd ChemoTreeVsDL
git checkout primenet

# Option A: full conda env (includes PyTorch CUDA 11.7)
conda env create -f environment.yml
conda activate flabnet_ml_pipeline_env

# Option B: venv + install PyTorch for YOUR GPU (see https://pytorch.org)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-colab.txt
# e.g. CUDA 12.x: pip install torch --index-url https://download.pytorch.org/whl/cu124

export PYTHONPATH="$(pwd)"
export PRIMENET_ROOT="$(pwd)/third_party/PrimeNet"
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Data

Put CSVs in `data/raw/` **or** copy prepared data from Colab:

- `mimic_cohort_NF_30_days.csv`
- `mimic_cohort_NF_30_days_admissions_labs_14_days.csv`

If you already have `MIMIC_IV/saved_data/` from Colab, copy it into the repo root and use `--skip-prepare`.

### 3. Run (recommended: shell wrapper)

```bash
chmod +x train_primenet_local.sh

# First time: prepare saved_data + train fold 0 (smoke test)
./train_primenet_local.sh --fold 0 --fast --prefix local_smoke

# Full training one fold (reuse prepared data + exported tensors)
./train_primenet_local.sh --fold 3 --skip-prepare --skip-export --prefix local_full

# All 5 folds, full training
./train_primenet_local.sh --all-folds --skip-prepare --prefix local_full

python scripts/summarize_folds.py --prefix local_full --model primenet
```

`colab_primenet_train.py` is the same driver on local and Colab; `train_primenet_local.sh` only sets `PYTHONPATH` and `PRIMENET_ROOT`.

### 4. Direct CLI (single fold, no wrapper)

```bash
python -m ts_model_training.main \
  --dataset MIMIC_IV \
  --cohort mimic_cohort_NF_30_days \
  --fold 0 \
  --model_type primenet \
  --feature_threshold \
  --grid none \
  --prefix primenet_run \
  --config_path config/ts_config_params.yaml
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
