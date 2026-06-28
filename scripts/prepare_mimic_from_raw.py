#!/usr/bin/env python3
"""
Prepare MIMIC_IV/saved_data from uploads in data/raw/ (Colab-friendly).

Input (either format in data/raw/):
  {cohort}.csv.gz   OR   {cohort}.csv
  {cohort}_admissions_labs_{days}_days.csv.gz   OR   ...csv

Output (always gzip under MIMIC_IV/saved_data/):
  cohorts/{cohort}.csv.gz
  features/{cohort}_admissions_labs_{days}_days.csv.gz
  processed_admission_features_for_ts/{cohort}/..._to_ts.csv.gz
  folds/{cohort}/fold_{0..4}.pkl
  top_features/mimic_top100_features.pkl

Or copy an existing Colab MIMIC_IV/saved_data/ tree into the repo root and use --skip-prepare.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.constants import PROJECT_ROOT, RAW_DATA_DIR
from utils.preprocessing_utils import split_cv_folds

ITEMIDS_TO_REMOVE = [50934, 50947, 51678]
SAVED_DATA = PROJECT_ROOT / "MIMIC_IV" / "saved_data"


def _find_raw_file(stem: str) -> Path:
    """Resolve data/raw/{stem}.csv.gz or .csv (prefer .gz)."""
    for name in (f"{stem}.csv.gz", f"{stem}.csv"):
        path = RAW_DATA_DIR / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing data/raw/{stem}.csv.gz (or .csv)\n"
        f"Upload gzip-compressed CSVs to data/raw/ when possible."
    )


def _write_gzip_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.name.endswith(".csv.gz"):
        df = pd.read_csv(src, compression="gzip")
    else:
        df = pd.read_csv(src)
    df.to_csv(dst, index=False, compression="gzip")


def prepare_cohort(cohort: str) -> None:
    src = _find_raw_file(cohort)
    _write_gzip_csv(src, SAVED_DATA / "cohorts" / f"{cohort}.csv.gz")


def prepare_features(cohort: str, days: int) -> None:
    name = f"{cohort}_admissions_labs_{days}_days"
    src = _find_raw_file(name)
    _write_gzip_csv(src, SAVED_DATA / "features" / f"{name}.csv.gz")


def build_to_ts(cohort: str, days: int) -> None:
    input_features = f"{cohort}_admissions_labs_{days}_days"
    features_path = SAVED_DATA / "features" / f"{input_features}.csv.gz"
    out_path = (
        SAVED_DATA / "processed_admission_features_for_ts" / cohort / f"{input_features}_to_ts.csv.gz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Skipping to_ts (exists): {out_path}")
        return

    labs = pd.read_csv(features_path, compression="gzip")
    labs["dischtime"] = pd.to_datetime(labs["dischtime"], errors="coerce")
    labs["date"] = pd.to_datetime(labs["date"], errors="coerce")
    labs = labs.dropna(subset=["dischtime", "date", "hadm_id", "itemid", "value"])
    labs = labs[~labs["itemid"].isin(ITEMIDS_TO_REMOVE)]
    labs["starttime"] = (labs["dischtime"] - pd.DateOffset(days=days)).apply(
        lambda x: x.replace(hour=0, minute=0, second=0)
    )
    labs["minute"] = (labs["date"] - labs["starttime"]).dt.total_seconds() / 60
    max_minute = (days + 1) * 24 * 60
    labs = labs[labs["minute"].between(0, max_minute)]
    labs["minute"] = labs["minute"].astype(np.int64)
    labs["hour"] = (labs["minute"] // 60).astype(np.int64)
    labs["day"] = (labs["hour"] // 24).astype(np.int64)
    labs[["subject_id", "hadm_id", "itemid", "value", "minute", "hour", "day"]].to_csv(
        out_path, index=False, compression="gzip"
    )
    print(f"Wrote {out_path}")


def build_top100(cohort: str, days: int) -> None:
    input_features = f"{cohort}_admissions_labs_{days}_days"
    cohort_df = pd.read_csv(SAVED_DATA / "cohorts" / f"{cohort}.csv.gz", compression="gzip")
    labs = pd.read_csv(SAVED_DATA / "features" / f"{input_features}.csv.gz", compression="gzip")
    labs = labs.merge(cohort_df[["hadm_id", "label"]], on="hadm_id")
    pos = labs[labs.label == 1].drop_duplicates(subset=["itemid", "hadm_id"])
    freq = (
        pos.groupby("itemid")["hadm_id"]
        .nunique()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    selected = freq.head(100)["itemid"].astype(str).tolist()
    out_dir = SAVED_DATA / "top_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mimic_top100_features.pkl", "wb") as f:
        pickle.dump(selected, f)
    print(f"Wrote top-100 features ({len(selected)} labs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="mimic_cohort_NF_30_days")
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepare_cohort(args.cohort)
    prepare_features(args.cohort, args.days)
    build_to_ts(args.cohort, args.days)
    split_cv_folds(args.cohort, 5, str(SAVED_DATA), seed=None)
    build_top100(args.cohort, args.days)
    print("Done:", SAVED_DATA)


if __name__ == "__main__":
    main()
