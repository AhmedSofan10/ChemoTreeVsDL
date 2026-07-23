# PrimeNet integration (native TimeBERT)

PrimeNet ([Chowdhury et al., AAAI 2023](https://doi.org/10.1609/aaai.v37i6.25876)) is integrated for **MIMIC-IV neutropenic fever and aplasia** using the same ChemoTree cohort construction, folds, and top-100 labs as STraTS.

For a full reproduction path, see [`REPRODUCE.md`](REPRODUCE.md). For Zenodo, see [`ZENODO.md`](ZENODO.md).

## Layout

| Path | Role |
|------|------|
| `ts_model_training/primenet/timebert/` | Native TimeBERT (`modules.py`, `models.py`, `collator.py`) |
| `ts_model_training/ts_models/ts_primenet.py` | `PRIMENET_TS` (parallel to `STRATS_TS`) |
| `ts_model_training/primenet/` | Snapshot builder, export, metrics, paths |
| `scripts/run_primenet_fig5.py` | Phased Fig. 5 orchestrator |
| `scripts/plot_fig5_heatmaps.py` | Paper-style heatmap PDF/PNG |
| `scripts/collect_primenet_results.py` | Aggregate fold metrics |
| `colab_primenet_fig5.py` / `colab_primenet_train.py` | Colab / local drivers |
| `notebooks/primenet_*.ipynb` | Chunked Colab workflows |

Attribution: `ts_model_training/primenet/timebert/ATTRIBUTION.md`.

## Train (local GPU)

```bash
cd ChemoTreeVsDL
git checkout primenet
export PYTHONPATH="$(pwd)"

conda env create -f environment.yml && conda activate flabnet_ml_pipeline_env
# or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-colab.txt

chmod +x train_primenet_local.sh run_primenet_fig5.sh

# Smoke
./train_primenet_local.sh --fold 0 --fast --prefix local_smoke

# Fig. 5 phases
./run_primenet_fig5.sh --phase pretrain-cohort --skip-prepare
./run_primenet_fig5.sh --phase finetune --scenarios nf --skip-prepare
./run_primenet_fig5.sh --phase collect --scenarios all
```

## Colab

Open notebooks on branch `primenet` (upload zips / HTTPS; no Drive required):

- `notebooks/primenet_nf_in_domain_colab.ipynb`
- `notebooks/primenet_mimic_iv_colab.ipynb`
- `notebooks/primenet_aplasia_finetune_colab.ipynb`

Optional T4 memory patch (idempotent):

```bash
python scripts/apply_colab_primenet_memory_fix.py
```

## Scope notes

- Default finetune uses `grid=none` (no nested HPO for PrimeNet yet).
- Fair comparison: same MIMIC-IV folds as STraTS — **not** vs PrimeNet paper MIMIC-III mortality AUROC.
- Keep `max_obs` / TimeBERT max length **512** identical between pretrain and finetune.
