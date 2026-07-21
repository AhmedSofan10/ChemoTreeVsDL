#!/usr/bin/env bash
#SBATCH --job-name=primenet_pt_cohort
#SBATCH --output=/home/hpc/iwbn/iwbn102h/ChemoTreeVsDL/logs/primenet_pt_cohort_%j.out
#SBATCH --error=/home/hpc/iwbn/iwbn102h/ChemoTreeVsDL/logs/primenet_pt_cohort_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#
# Job 1: NF cohort SSL pretrain (~30 min on cluster).
# Usage: sbatch scripts/hpc/primenet_pretrain_cohort.sh
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

./run_primenet_fig5.sh --phase pretrain-cohort --skip-prepare "$@"
