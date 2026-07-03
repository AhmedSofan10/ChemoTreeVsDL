#!/usr/bin/env bash
#SBATCH --job-name=primenet_pt_cohort
#SBATCH --output=logs/primenet_pt_cohort_%j.out
#SBATCH --error=logs/primenet_pt_cohort_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# Job 1: NF cohort SSL pretrain (~30 min on cluster).
# Usage: sbatch scripts/hpc/primenet_pretrain_cohort.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

./run_primenet_fig5.sh --phase pretrain-cohort --skip-prepare "$@"
