#!/usr/bin/env python3
"""
Build aplasia finetune artifacts by filtering existing MIMIC-IV mimic_all labs.

Does NOT touch mimic_top100_features.pkl (keep the vocab used by mimic_all / NF
pretrain checkpoints).

Writes under MIMIC_IV/saved_data/:
  cohorts/mimic_cohort_aplasia_45_days.csv.gz
  processed_admission_features_for_ts/mimic_cohort_aplasia_45_days/
      mimic_cohort_aplasia_45_days_admissions_labs_14_days_to_ts.csv.gz
  folds/mimic_cohort_aplasia_45_days/fold_{0..4}.pkl

Example:
  python scripts/prepare_aplasia_from_mimic_all.py \\
    --cohort-csv ~/Downloads/mimic_cohort_aplasia_45_days.csv.gz
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.constants import PROJECT_ROOT
from utils.preprocessing_utils import split_cv_folds

COHORT = "mimic_cohort_aplasia_45_days"
DAYS = 14
SAVED = PROJECT_ROOT / "MIMIC_IV" / "saved_data"
SOURCE_TO_TS = (
    SAVED
    / "processed_admission_features_for_ts"
    / "mimic_all"
    / f"mimic_all_admissions_labs_{DAYS}_days_to_ts.csv.gz"
)


def _place_cohort(src: Path) -> Path:
    dst = SAVED / "cohorts" / f"{COHORT}.csv.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffixes[-2:] == [".csv", ".gz"] or str(src).endswith(".csv.gz"):
        shutil.copy2(src, dst)
    else:
        pd.read_csv(src).to_csv(dst, index=False, compression="gzip")
    print(f"Cohort → {dst}")
    return dst


def _filter_labs(cohort_path: Path, chunksize: int) -> tuple[Path, set[int]]:
    if not SOURCE_TO_TS.is_file():
        raise FileNotFoundError(
            f"Missing full mimic_all labs:\n  {SOURCE_TO_TS}\n"
            "Need MIMIC-IV mimic_all extraction first."
        )

    cohort = pd.read_csv(cohort_path, compression="gzip")
    hadm_ids = set(int(h) for h in cohort["hadm_id"].unique())
    print(f"Aplasia admissions: {len(hadm_ids):,}")

    out = (
        SAVED
        / "processed_admission_features_for_ts"
        / COHORT
        / f"{COHORT}_admissions_labs_{DAYS}_days_to_ts.csv.gz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()

    kept_rows = 0
    found_hadm: set[int] = set()
    first = True
    for chunk in pd.read_csv(SOURCE_TO_TS, chunksize=chunksize):
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if chunk.empty:
            continue
        if "hour" not in chunk.columns:
            chunk["hour"] = (chunk["minute"] // 60).astype("int64")
        if "day" not in chunk.columns:
            chunk["day"] = (chunk["hour"] // 24).astype("int64")
        cols = ["subject_id", "hadm_id", "itemid", "value", "minute", "hour", "day"]
        chunk[cols].to_csv(
            out, index=False, compression="gzip", mode="a", header=first
        )
        first = False
        kept_rows += len(chunk)
        found_hadm.update(int(x) for x in chunk["hadm_id"].unique().tolist())
        print(f"  kept rows so far: {kept_rows:,}  hadm: {len(found_hadm):,}", flush=True)

    if kept_rows == 0:
        raise RuntimeError("No lab rows matched aplasia hadm_ids in mimic_all labs")

    missing = hadm_ids - found_hadm
    print(f"Wrote {kept_rows:,} lab rows → {out}")
    print(
        f"Coverage: {len(found_hadm)}/{len(hadm_ids)} admissions "
        f"({100.0 * len(found_hadm) / len(hadm_ids):.2f}%)"
    )
    if missing:
        print(f"Dropping {len(missing)} aplasia admissions with no labs from cohort")
        cohort = cohort[cohort["hadm_id"].astype(int).isin(found_hadm)]
        cohort.to_csv(cohort_path, index=False, compression="gzip")
        print(f"Updated cohort → {cohort_path} ({len(cohort)} rows)")
    return out, found_hadm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=Path.home() / "Downloads" / "mimic_cohort_aplasia_45_days.csv.gz",
        help="Path to aplasia cohort CSV (.csv or .csv.gz)",
    )
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument(
        "--skip-folds",
        action="store_true",
        help="Only write cohort + labs (reuse existing folds)",
    )
    args = parser.parse_args()

    if not args.cohort_csv.is_file():
        raise FileNotFoundError(args.cohort_csv)

    cohort_path = _place_cohort(args.cohort_csv)
    _filter_labs(cohort_path, args.chunksize)

    if not args.skip_folds:
        split_cv_folds(COHORT, 5, str(SAVED), seed=None)

    top100 = SAVED / "top_features" / "mimic_top100_features.pkl"
    if top100.is_file():
        print(f"Kept existing top-100 (for mimic_all ckpt transfer): {top100}")
    else:
        print(f"WARNING: missing {top100} — needed to load mimic_all checkpoint")

    print("Done. Ready to finetune with --cohort", COHORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
