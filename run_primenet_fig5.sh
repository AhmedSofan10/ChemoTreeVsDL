#!/usr/bin/env bash
# Run STraTS Fig.5-style PrimeNet scenarios on MIMIC-IV NF (5 folds each).
#
# HPC (split jobs, 24h limit):
#   sbatch scripts/hpc/primenet_pretrain_cohort.sh
#   sbatch scripts/hpc/primenet_pretrain_mimicall.sh
#   sbatch scripts/hpc/primenet_finetune_nf.sh
#   sbatch scripts/hpc/primenet_finetune_mimicall.sh
#
# Examples:
#   ./run_primenet_fig5.sh --phase pretrain-cohort --skip-prepare
#   ./run_primenet_fig5.sh --phase finetune --skip-prepare --scenarios nf
#   ./run_primenet_fig5.sh --fast --phase finetune --scenarios none_none
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec python3 scripts/run_primenet_fig5.py "$@"
