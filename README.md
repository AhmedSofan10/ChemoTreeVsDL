# ChemoTreeVsDL + PrimeNet (`primenet` branch)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: branch](https://img.shields.io/badge/branch-primenet-informational)](https://github.com/AhmedSofan10/ChemoTreeVsDL/tree/primenet)

**Fork of [bionetslab/ChemoTreeVsDL](https://github.com/bionetslab/ChemoTreeVsDL) (FLabBench / ChemoTree)** with a **native PrimeNet (TimeBERT)** integration for irregular laboratory time series.

This branch supports **STraTS Fig. 5–style** pretrain → finetune experiments on **MIMIC-IV** neutropenic fever (NF) and aplasia:

| Pretrain | Finetune | Meaning |
|----------|----------|---------|
| `cohort` | `all` / `final` | SSL on the task cohort, then full or frozen-backbone finetune |
| `mimic_all` | `all` / `final` | SSL on all suitable MIMIC admissions, then finetune on the task |
| `none` | `none` / `final` | Supervised only (optional / incomplete in some runs) |

> **Paper companion.** Results for the project report were produced with the scripts and notebooks below. Heatmaps in [`figures/`](figures/) can be regenerated with [`scripts/plot_fig5_heatmaps.py`](scripts/plot_fig5_heatmaps.py).

---

## Quick start

```bash
git clone -b primenet https://github.com/AhmedSofan10/ChemoTreeVsDL.git
cd ChemoTreeVsDL

# Option A — conda (full stack, includes classical ML)
conda env create -f environment.yml
conda activate flabnet_ml_pipeline_env

# Option B — Colab / lightweight venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-colab.txt

export PYTHONPATH="$(pwd)"
```

**Data (required):** MIMIC-IV is credentialed. You need either:

1. Prepared tree under `MIMIC_IV/saved_data/` (cohorts, folds, `*_to_ts.csv.gz`, top-100 features), **or**
2. Raw cohort + labs CSVs in `data/raw/` (see [docs/REPRODUCE.md](docs/REPRODUCE.md)).

---

## Reproduce Fig. 5–style PrimeNet runs

### Recommended: phased CLI

```bash
# 1) Prepare NF saved_data (skip if you already have it)
python scripts/run_primenet_fig5.py --phase prepare

# 2) SSL pretrain on NF cohort
python scripts/run_primenet_fig5.py --phase pretrain-cohort --skip-prepare

# 3) Finetune NF (cohort/all + cohort/final)
python scripts/run_primenet_fig5.py --phase finetune --scenarios nf --skip-prepare

# 4) Optional: mimic_all pretrain + finetune (needs mimic_all labs)
python scripts/run_primenet_fig5.py --phase pretrain-mimicall --skip-prepare
python scripts/run_primenet_fig5.py --phase finetune --scenarios mimicall --skip-prepare

# 5) Collect mean ± std table
python scripts/run_primenet_fig5.py --phase collect --scenarios all
```

Wrapper: `./run_primenet_fig5.sh …` (same arguments).

### Colab notebooks (chunked; no Google Drive required)

| Notebook | Purpose |
|----------|---------|
| [`notebooks/primenet_nf_in_domain_colab.ipynb`](notebooks/primenet_nf_in_domain_colab.ipynb) | NF SSL + `cohort/all` + `cohort/final` |
| [`notebooks/primenet_mimic_iv_colab.ipynb`](notebooks/primenet_mimic_iv_colab.ipynb) | `mimic_all` → NF finetune |
| [`notebooks/primenet_aplasia_finetune_colab.ipynb`](notebooks/primenet_aplasia_finetune_colab.ipynb) | Aplasia finetune (+ optional in-domain) |
| [`notebooks/strats_nf_in_domain_colab.ipynb`](notebooks/strats_nf_in_domain_colab.ipynb) | STraTS NF in-domain (comparison) |

### Plot paper-style heatmaps

```bash
pip install matplotlib numpy   # if not already installed
python scripts/plot_fig5_heatmaps.py --out-dir figures
```

Outputs: `figures/fig_primenet_fig5.pdf`, `figures/fig_strats_fig5.pdf` (+ PNG).

### Summarize folds

```bash
python scripts/collect_primenet_results.py --prefix fig5_cohort_all --fig5-row
python scripts/summarize_folds.py --prefix fig5_cohort_all --model primenet
```

---

## Repository layout

```text
ChemoTreeVsDL/
├── README.md                 ← you are here
├── LICENSE                   ← MIT (this fork’s packaging + PrimeNet integration)
├── CITATION.cff              ← cite this software
├── environment.yml           ← full conda env
├── requirements-colab.txt    ← minimal GPU / Colab deps
├── config/                   ← YAML + path constants
├── ts_model_training/        ← DL trainers (STraTS, PrimeNet, …)
│   └── primenet/             ← native TimeBERT + adapters
├── ml_model_training/        ← classical ML (CatBoost, …)
├── scripts/                  ← prepare / Fig.5 / collect / plot / HPC
├── notebooks/                ← Colab entry points
├── figures/                  ← regenerated Fig.5-style heatmaps
├── docs/
│   ├── PRIMENET.md           ← architecture & training details
│   ├── REPRODUCE.md          ← end-to-end reproduction guide
│   └── ZENODO.md             ← how to mint a DOI
├── MIMIC_IV/                 ← cohort notebooks + saved_data/ (gitignored)
├── UKEr/                     ← UKEr cohort notebooks
└── data/raw/                 ← place credentialed CSVs here (gitignored)
```

Upstream classical + STraTS training still works via `train_ml_models.sh` / `train_ts_models.sh` (see original FLabBench README content below).

---

## Models (full FLabBench set)

**Classical:** RF, GB, XGB, CatBoost, LR, MLP  

**Temporal:** GRU, LSTM, TCN, SAnD, GRU-D, InterpNet, STraTS, **PrimeNet (TimeBERT)**

---

## Cohorts (ChemoTree protocol)

- Cancer + chemotherapy admissions  
- **Aplasia:** transfusion / low ANC (45-day post-discharge window)  
- **Neutropenic fever:** neutropenia + fever (30-day window)  
- **Observation:** 14 days before discharge; top-100 labs + age/gender; patient-stratified 5-fold CV  

---

## Classical / upstream usage

```bash
conda env create -f environment.yml && conda activate flabnet_ml_pipeline_env
bash train_ml_models.sh mimic full_run    # classical
bash train_ts_models.sh                   # temporal (prepared inputs)
```

---

## Citation

If you use this fork / PrimeNet integration:

```bibtex
@software{sofan2026primenet_chemotree,
  author  = {Sofan, Ahmed},
  title   = {ChemoTreeVsDL with PrimeNet (TimeBERT) for MIMIC-IV NF and aplasia},
  year    = {2026},
  url     = {https://github.com/AhmedSofan10/ChemoTreeVsDL},
  version = {primenet},
  note    = {Fork of bionetslab/ChemoTreeVsDL (FLabBench)}
}
```

Please also cite the ChemoTree / FLabBench paper and PrimeNet:

- Rahimi et al., *Non-temporal tree-based models…*, medRxiv 2025. [doi:10.64898/2025.12.12.25342142](https://doi.org/10.64898/2025.12.12.25342142)  
- Chowdhury et al., *PrimeNet*, AAAI 2023. [doi:10.1609/aaai.v37i6.25876](https://doi.org/10.1609/aaai.v37i6.25876)  
- Tipirneni & Reddy, *STraTS*, arXiv:2207.00227  

See [`CITATION.cff`](CITATION.cff) and [`docs/ZENODO.md`](docs/ZENODO.md) for DOI minting.

---

## Attribution & license

- **Upstream:** [bionetslab/ChemoTreeVsDL](https://github.com/bionetslab/ChemoTreeVsDL) (FLabBench), Biomedical Network Science Lab, FAU.  
- **PrimeNet / TimeBERT:** adapted from [ranakroychowdhury/PrimeNet](https://github.com/ranakroychowdhury/PrimeNet); see `ts_model_training/primenet/timebert/ATTRIBUTION.md`.  
- **This branch:** MIT License for packaging, orchestration, and integration code added on `primenet` (see [`LICENSE`](LICENSE)).

---

## Support

Open an issue on GitHub for bugs or reproduction problems. For MIMIC-IV access, follow [PhysioNet](https://physionet.org/content/mimiciv/).
