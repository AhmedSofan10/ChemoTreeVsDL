#!/usr/bin/env bash
#SBATCH --job-name=primenet_pt_mimicall
#SBATCH --output=/home/hpc/iwbn/iwbn102h/ChemoTreeVsDL/logs/primenet_pt_mimicall_%j.out
#SBATCH --error=/home/hpc/iwbn/iwbn102h/ChemoTreeVsDL/logs/primenet_pt_mimicall_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#
# Job 2: full mimic_all SSL pretrain.
# Extracts mimic_all from raw MIMIC only if labs are missing (idempotent),
# otherwise reuses MIMIC_IV/saved_data/. For a smaller test run, pass
# --mimic-all-cohort mimic_all_10pct (or mimic_all_2pct).
#
# Usage: sbatch scripts/hpc/primenet_pretrain_mimicall.sh
#        sbatch scripts/hpc/primenet_pretrain_mimicall.sh --mimic-all-cohort mimic_all_10pct
#
set -euo pipefail
ROOT="${CHEMOTREEVSDL_ROOT:-/home/hpc/iwbn/iwbn102h/ChemoTreeVsDL}"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# --- Conda environment (adjust CONDA_BASE / module load to your cluster) ---
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate flabnet_ml_pipeline_env

# --- Raw MIMIC-IV location (only used if mimic_all is not yet extracted) ---
# Point this at the dir containing hosp/ (keep a trailing slash), or export
# MIMIC_DIR before sbatch. Extraction is skipped automatically if labs exist.
export MIMIC_DIR="${MIMIC_DIR:-/home/jovyan/data_common/mimiciv/}"

# Full mimic_all SSL pretrain. Extraction runs from raw MIMIC only if the
# mimic_all labs are missing under MIMIC_IV/saved_data/ (idempotent).
./run_primenet_fig5.sh \
  --phase pretrain-mimicall \
  --skip-prepare \
  --mimic-all-cohort mimic_all \
  "$@"
