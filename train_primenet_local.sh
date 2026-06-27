#!/usr/bin/env bash
# Train PrimeNet locally (same entry as Colab, no notebook needed).
#
# Usage examples:
#   ./train_primenet_local.sh --fold 0 --fast
#   ./train_primenet_local.sh --fold 3 --skip-prepare --skip-export --prefix local_full
#   ./train_primenet_local.sh --all-folds --skip-prepare --prefix local_full
#   ./train_primenet_local.sh --prepare-only
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

exec python colab_primenet_train.py "$@"
