#!/usr/bin/env bash
#SBATCH --job-name=primenet_ft_mimic
#SBATCH --output=logs/primenet_ft_mimic_%j.out
#SBATCH --error=logs/primenet_ft_mimic_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# Job 4: mimicall finetune (2 scenarios x 5 folds).
# Requires fig5_pt_mimicall checkpoint from Job 2 (or full mimic_all pretrain).
#
# Usage: sbatch scripts/hpc/primenet_finetune_mimicall.sh
#        sbatch scripts/hpc/primenet_finetune_mimicall.sh --mimic-all-cohort mimic_all
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# --- Conda environment (adjust CONDA_BASE / module load to your cluster) ---
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate flabnet_ml_pipeline_env

./run_primenet_fig5.sh \
  --phase finetune \
  --skip-prepare \
  --scenarios mimicall \
  --mimic-all-cohort mimic_all \
  "$@"
