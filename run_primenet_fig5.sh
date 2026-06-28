#!/usr/bin/env bash
# Run all STraTS Fig.5-style PrimeNet scenarios on MIMIC-IV NF (5 folds each).
#
# Examples:
#   ./run_primenet_fig5.sh --skip-prepare
#   ./run_primenet_fig5.sh --fast --scenarios cohort_all,none_none
#   ./run_primenet_fig5.sh --skip-prepare --skip-extract-mimic-all --scenarios mimicall_all
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec python3 scripts/run_primenet_fig5.py "$@"
