"""
Export MIMIC-IV cohort lab series to PrimeNet tensor layout [N, T, 2*D+1].

Uses snapshot_builder (shared with PreprocessorD).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from config.constants import PRIMENET_DATA_DIR
from ts_model_training.primenet.snapshot_builder import build_fold_tensors
from ts_model_training.utils import load_fold_file, set_all_paths


def _load_tables(args):
    import pickle as pkl

    paths = set_all_paths(args, out=False)
    args.paths = paths
    import pandas as pd

    cohort = pd.read_csv(paths["cohort_path"], compression="gzip")
    dtype_spec = {
        "subject_id": "int64",
        "hadm_id": "int64",
        "minute": "int64",
        "itemid": "string",
        "value": "float64",
    }
    data = pd.read_csv(paths["data_path"], dtype=dtype_spec)
    with open(paths["feature_path"] / "mimic_top100_features.pkl", "rb") as f:
        features = [str(x) for x in pkl.load(f)]
    data = data[data["itemid"].isin(features)]
    return cohort, data, features


def export_fold(
    cohort: str,
    fold: int,
    max_obs: int = 512,
    days_before_discharge: int = 14,
    dataset: str = "MIMIC_IV",
) -> Path:
    args = argparse.Namespace(
        dataset=dataset,
        cohort=cohort,
        fold=fold,
        days_before_discharge=days_before_discharge,
        split_seed=None,
    )
    args.paths = set_all_paths(args, out=False)
    train_ids, val_ids, test_ids = load_fold_file(args)
    cohort_df, data, features = _load_tables(args)
    packs = build_fold_tensors(
        cohort_df,
        data,
        features,
        train_ids,
        val_ids,
        test_ids,
        max_obs,
        days_before_discharge,
    )

    out_dir = PRIMENET_DATA_DIR / cohort / f"fold_{fold}"
    for split_name in ("finetune", "pretrain"):
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in packs[split_name].items():
            torch.save(torch.tensor(arr), split_dir / f"{name}.pt")

    n_feat = packs["meta"]["input_dim"]
    meta = {
        "cohort": cohort,
        "fold": fold,
        "n_features": n_feat,
        "shapes": {k: list(v.shape) for k, v in packs["finetune"].items() if k.startswith("X")},
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Exported PrimeNet tensors -> {out_dir} (D={n_feat}, {meta['shapes']})")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Export MIMIC-IV fold to PrimeNet .pt tensors")
    parser.add_argument("--cohort", default="mimic_cohort_NF_30_days")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-obs", type=int, default=512)
    parser.add_argument("--days-before-discharge", type=int, default=14)
    args = parser.parse_args()
    export_fold(args.cohort, args.fold, args.max_obs, args.days_before_discharge)


if __name__ == "__main__":
    main()
