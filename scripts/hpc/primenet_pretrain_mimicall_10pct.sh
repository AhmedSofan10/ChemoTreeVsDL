#!/usr/bin/env bash
#SBATCH --job-name=primenet_pt_mimic10
#SBATCH --output=logs/primenet_pt_mimic10_%j.out
#SBATCH --error=logs/primenet_pt_mimic10_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#
# Job 2: mimic_all 10% SSL pretrain (pipeline validation).
# Requires full mimic_all labs under MIMIC_IV/saved_data/.
# Supervisor runs full mimic_all with --mimic-all-cohort mimic_all.
#
# Usage: sbatch scripts/hpc/primenet_pretrain_mimicall_10pct.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

./run_primenet_fig5.sh \
  --phase pretrain-mimicall \
  --skip-prepare \
  --mimic-all-cohort mimic_all_10pct \
  "$@"
