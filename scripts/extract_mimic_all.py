#!/usr/bin/env python3
"""Build mimic_all pretrain artifacts from raw MIMIC-IV (STraTS / PrimeNet)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ts_model_training.extractor import ExtractorPretrain
from ts_model_training.logger import Logger


class Args:
    dataset = "MIMIC_IV"
    cohort = "mimic_all"
    days_before_discharge = 14
    feature_threshold = False
    logger = Logger(None)


def main() -> None:
    out = (
        _ROOT
        / "MIMIC_IV"
        / "saved_data"
        / "processed_admission_features_for_ts"
        / "mimic_all"
        / "mimic_all_admissions_labs_14_days_to_ts.csv.gz"
    )
    if out.is_file():
        print(f"Already exists: {out}")
        return
    print("Starting mimic_all extraction (labevents ~2.5GB — expect 1–3+ hours)...")
    ExtractorPretrain(Args())
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
