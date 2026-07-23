# Reproduce experiments end-to-end

This guide walks from a clean clone to Fig. 5–style metrics and heatmaps.

## 0. Prerequisites

| Item | Notes |
|------|--------|
| Python 3.10+ | Conda env or venv |
| GPU | Strongly recommended (Colab T4 works for chunked runs) |
| MIMIC-IV access | [PhysioNet credentialed](https://physionet.org/content/mimiciv/) |
| Disk | Several GB for `saved_data` + checkpoints |

```bash
git clone -b primenet https://github.com/AhmedSofan10/ChemoTreeVsDL.git
cd ChemoTreeVsDL
export PYTHONPATH="$(pwd)"
```

## 1. Environment

**Full (classical + DL):**

```bash
conda env create -f environment.yml
conda activate flabnet_ml_pipeline_env
```

**Minimal (PrimeNet / STraTS on Colab):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-colab.txt
```

Check GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

## 2. Data layout

### Option A — Use prepared `saved_data` (fastest)

Place a tree like:

```text
MIMIC_IV/saved_data/
  cohorts/mimic_cohort_NF_30_days.csv.gz
  folds/mimic_cohort_NF_30_days/fold_{0..4}.pkl
  top_features/mimic_top100_features.pkl
  processed_admission_features_for_ts/..._to_ts.csv.gz
  # optional for transfer:
  cohorts/mimic_all*.csv.gz
  processed_admission_features_for_ts/mimic_all*_to_ts.csv.gz
```

Then always pass `--skip-prepare` to Fig. 5 scripts.

### Option B — Build from raw CSVs

Put in `data/raw/`:

- `mimic_cohort_NF_30_days.csv`
- `mimic_cohort_NF_30_days_admissions_labs_14_days.csv`

```bash
python scripts/prepare_mimic_from_raw.py
# or
python scripts/run_primenet_fig5.py --phase prepare
```

### Aplasia

Use `scripts/prepare_aplasia_from_mimic_all.py` to filter mimic_all labs to aplasia `hadm_id`s when you already have mimic_all `*_to_ts` files. See `notebooks/primenet_aplasia_finetune_colab.ipynb`.

## 3. Train (Fig. 5 matrix)

### Smoke test (1 fold, fast)

```bash
python scripts/run_primenet_fig5.py --phase finetune --scenarios none_none --fast --skip-prepare
```

### NF in-domain (recommended report path)

```bash
python scripts/run_primenet_fig5.py --phase pretrain-cohort --skip-prepare
python scripts/run_primenet_fig5.py --phase finetune --scenarios nf --skip-prepare
python scripts/collect_primenet_results.py --prefix fig5_cohort_all --fig5-row
python scripts/collect_primenet_results.py --prefix fig5_cohort_final --fig5-row
```

### mimic_all → NF transfer

```bash
python scripts/run_primenet_fig5.py --phase pretrain-mimicall --skip-prepare \
  --mimic-all-cohort mimic_all
python scripts/run_primenet_fig5.py --phase finetune --scenarios mimicall --skip-prepare \
  --mimic-all-cohort mimic_all
```

### Colab (chunked)

Open the matching notebook under `notebooks/`, upload `*_saved_data.zip` / progress zips as documented in the notebook cells, run one chunk per session, download progress zips.

### STraTS comparison

```bash
# Prefer notebooks/strats_nf_in_domain_colab.ipynb
# or train via ts_model_training.main with --model_type strats and ts_config_params_best.yaml
```

## 4. Collect metrics & plot

```bash
python scripts/run_primenet_fig5.py --phase collect --scenarios all
python scripts/plot_fig5_heatmaps.py --out-dir figures
```

Edit numeric matrices inside `scripts/plot_fig5_heatmaps.py` if you recompute means/stds.

## 5. Where results live

```text
MIMIC_IV/saved_data/results/<cohort>/time_series/
  pretrain/primenet/fig5_pt_*/
  finetune/primenet/fig5_cohort_all/fold_*/grid_none/results_final.csv
  finetune/strats/...
```

Test metrics: last `split=test` row in each `results_final.csv`.

## 6. HPC

See `scripts/hpc/*.sh` (SLURM). Prefer one phase per job (24h limits).

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `pandas` StringDtype pickle error | `pip install 'pandas>=2.3'` |
| Positional embedding size mismatch | Keep `max_obs=512` identical between pretrain and finetune |
| Colab OOM on pretrain | Run `python scripts/apply_colab_primenet_memory_fix.py` once, reduce batch / use chunked notebooks |
| `itemid` int vs str join error | Pull latest `primenet` (preprocessor coerces to str) |
| Missing checkpoint | Ensure `fig5_pt_* /fold_0/grid_none/checkpoint_best.bin` exists before finetune |

## 8. What is *not* redistributed

- Raw MIMIC-IV files  
- Full `saved_data/` and checkpoints (too large / DUA-bound)  
- UKEr private data  

You must regenerate artifacts under your PhysioNet agreement.
