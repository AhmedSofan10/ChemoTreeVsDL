#!/usr/bin/env bash
#SBATCH --job-name=primenet_ft_nf
#SBATCH --output=logs/primenet_ft_nf_%j.out
#SBATCH --error=logs/primenet_ft_nf_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# Job 3: NF finetune (cohort + none scenarios, 4 x 5 folds).
# Requires fig5_pt_cohort checkpoint from Job 1.
#
# Usage: sbatch scripts/hpc/primenet_finetune_nf.sh
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
  --scenarios nf \
  "$@"
