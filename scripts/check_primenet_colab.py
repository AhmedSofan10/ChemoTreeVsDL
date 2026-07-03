#!/usr/bin/env python3
"""Verify MIMIC_IV/saved_data is ready for PrimeNet Fig.5 on Colab."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.constants import PROJECT_ROOT

DATASET = "MIMIC_IV"
NF = "mimic_cohort_NF_30_days"
SAVED = PROJECT_ROOT / DATASET / "saved_data"
PT_PREFIX = "fig5_pt_cohort"


def _ckpt_dir() -> Path:
    return (
        SAVED
        / "results"
        / NF
        / "time_series"
        / "pretrain"
        / "primenet"
        / PT_PREFIX
        / "fold_0"
        / "grid_none"
    )


def main() -> int:
    required = [
        ("cohort", SAVED / "cohorts" / f"{NF}.csv.gz"),
        ("labs to_ts", SAVED / "processed_admission_features_for_ts" / NF / f"{NF}_admissions_labs_14_days_to_ts.csv.gz"),
        ("top-100 features", SAVED / "top_features" / "mimic_top100_features.pkl"),
    ]
    for fold in range(5):
        required.append((f"fold {fold}", SAVED / "folds" / NF / f"fold_{fold}.pkl"))

    missing = [label for label, path in required if not path.is_file()]
    if missing:
        print("MISSING data (run cell 5a with SKIP_PREPARE=False, or upload full saved_data):")
        for label in missing:
            print(f"  - {label}")
        print("\nIf you only uploaded 2 CSVs: set SKIP_PREPARE=False and run cell 5a first.")
        return 1

    ckpt = _ckpt_dir()
    ckpt_files = [
        ("pretrain checkpoint", ckpt / "checkpoint_best.bin"),
        ("pretrain variables", ckpt / "primenet_saved_variables.pkl"),
    ]
    ckpt_missing = [label for label, path in ckpt_files if not path.is_file()]
    if ckpt_missing:
        print("Data OK, but pretrain checkpoint missing (needed for cohort_all / cohort_final):")
        for label in ckpt_missing:
            print(f"  - {label}")
        print(f"\nRun cell 5a first, or copy HPC folder:\n  {ckpt}")
        print("\nYou can still test with: --scenarios none_none (no checkpoint needed)")
        return 2

    print("OK — data + pretrain checkpoint ready for full NF finetune.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
