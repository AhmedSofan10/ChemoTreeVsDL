#!/usr/bin/env python3
"""
Build a smaller mimic_all cohort for pipeline testing (e.g. 10% of admissions).

Reads the full mimic_all artifacts under MIMIC_IV/saved_data/ and writes:
  cohorts/{target}.csv.gz
  processed_admission_features_for_ts/{target}/..._to_ts.csv.gz
  folds/{target}/fold_0.pkl  (80/20 train/val admission split, empty test)

Example:
  python scripts/subset_mimic_all_cohort.py --frac 0.1 --target mimic_all_10pct
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

from config.constants import PROJECT_ROOT

SAVED = PROJECT_ROOT / "MIMIC_IV" / "saved_data"
SOURCE = "mimic_all"
DAYS = 14


def _paths(cohort: str) -> dict[str, Path]:
    stem = f"{cohort}_admissions_labs_{DAYS}_days"
    return {
        "cohort": SAVED / "cohorts" / f"{cohort}.csv.gz",
        "to_ts": SAVED / "processed_admission_features_for_ts" / cohort / f"{stem}_to_ts.csv.gz",
        "folds": SAVED / "folds" / cohort,
    }


def _sample_hadm_ids(cohort_df: pd.DataFrame, frac: float, seed: int) -> set[int]:
    hadm_ids = cohort_df["hadm_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(len(hadm_ids) * frac)))
    chosen = rng.choice(hadm_ids, size=n_keep, replace=False)
    return set(int(h) for h in chosen)


def _write_subset_cohort(source_df: pd.DataFrame, hadm_ids: set[int], out: Path) -> pd.DataFrame:
    sub = source_df[source_df["hadm_id"].isin(hadm_ids)].copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, compression="gzip")
    return sub


def _filter_to_ts(source_to_ts: Path, hadm_ids: set[int], out: Path, chunksize: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()
    first = True
    kept = 0
    for chunk in pd.read_csv(source_to_ts, chunksize=chunksize):
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if chunk.empty:
            continue
        kept += len(chunk)
        chunk.to_csv(out, index=False, compression="gzip", mode="a", header=first)
        first = False
    if kept == 0:
        raise RuntimeError(f"No lab rows kept for subset; check {source_to_ts}")
    print(f"Wrote {kept:,} lab rows → {out}")


def _write_fold_0(cohort_df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    """80/20 admission split (matches full mimic_all pretrain fold ratio)."""
    hadm = cohort_df[["subject_id", "hadm_id"]].drop_duplicates()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(hadm))
    n_val = max(1, int(round(0.2 * len(hadm))))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    train_hadms = np.asarray(hadm.iloc[train_idx][["subject_id", "hadm_id"]])
    val_hadms = np.asarray(hadm.iloc[val_idx][["subject_id", "hadm_id"]])
    test_hadms = np.empty((0, 2), dtype=train_hadms.dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fold_0.pkl", "wb") as f:
        pickle.dump([train_hadms, val_hadms, test_hadms], f)
    print(f"fold_0: train={len(train_hadms):,}, val={len(val_hadms):,}, test=0")


def main() -> int:
    parser = argparse.ArgumentParser(description="Subset mimic_all for PrimeNet pipeline tests")
    parser.add_argument("--frac", type=float, default=0.1, help="Fraction of admissions to keep (default 0.1)")
    parser.add_argument("--target", default="mimic_all_10pct", help="Output cohort name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--force", action="store_true", help="Overwrite existing subset artifacts")
    args = parser.parse_args()

    if not 0 < args.frac <= 1:
        raise SystemExit("--frac must be in (0, 1]")

    src = _paths(SOURCE)
    dst = _paths(args.target)
    for key, path in dst.items():
        if key == "folds":
            continue
        if path.exists() and not args.force:
            print(f"[skip] exists: {path}")
            return 0

    if not src["cohort"].is_file():
        raise SystemExit(f"Missing source cohort: {src['cohort']}")
    if not src["to_ts"].is_file():
        raise SystemExit(f"Missing source labs: {src['to_ts']}")

    cohort_df = pd.read_csv(src["cohort"], compression="gzip")
    hadm_ids = _sample_hadm_ids(cohort_df, args.frac, args.seed)
    print(f"Sampling {len(hadm_ids):,} / {cohort_df['hadm_id'].nunique():,} admissions ({args.frac:.0%})")

    sub_cohort = _write_subset_cohort(cohort_df, hadm_ids, dst["cohort"])
    print(f"Wrote cohort → {dst['cohort']} ({len(sub_cohort):,} rows)")

    _filter_to_ts(src["to_ts"], hadm_ids, dst["to_ts"], args.chunksize)
    _write_fold_0(sub_cohort, dst["folds"], args.seed)
    print(f"Done. Use --cohort {args.target} for pretrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
