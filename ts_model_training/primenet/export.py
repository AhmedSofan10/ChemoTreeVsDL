"""
Export MIMIC-IV cohort lab series to PrimeNet tensor layout [N, T, 2*D+1].

Uses the same CSVs and CV folds as STraTS (mimic_cohort_NF_30_days, 14-day window).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config.constants import PRIMENET_DATA_DIR
from ts_model_training.utils import (
    compute_means_stds_df,
    ids_in_data,
    load_fold_file,
    remove_features_not_in_train,
    set_all_paths,
)


def _load_tables(args) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    paths = set_all_paths(args, out=False)
    args.paths = paths
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
        features = [str(x) for x in pickle.load(f)]
    data = data[data["itemid"].isin(features)]
    return cohort, data, features


def _build_sequences(
    data: pd.DataFrame,
    hadm_ids: np.ndarray,
    features: List[str],
    means_stds: pd.DataFrame,
    label_map: Dict[int, int],
    max_obs: int,
    max_minutes: float,
) -> Tuple[np.ndarray, np.ndarray]:
    var_to_ind = {v: i for i, v in enumerate(features)}
    d = len(features)
    tensors = []
    labels = []

    sub = data.loc[data.hadm_id.isin(hadm_ids)].copy()
    sub = sub.merge(means_stds, on="itemid", how="left")
    sub["value"] = (sub["value"] - sub["mean"]) / sub["std"]
    sub = sub.groupby(["hadm_id", "minute", "itemid"]).value.mean().reset_index()

    for hadm_id in hadm_ids:
        grp = sub.loc[sub.hadm_id == hadm_id]
        if grp.empty:
            continue
        times = sorted(grp["minute"].unique())
        if len(times) > max_obs:
            rng = np.random.default_rng(int(hadm_id) % (2**31))
            times = sorted(rng.choice(times, size=max_obs, replace=False))
        rows = []
        for t in times:
            t_grp = grp.loc[grp.minute == t]
            vals = np.zeros(d, dtype=np.float32)
            mask = np.zeros(d, dtype=np.float32)
            for row in t_grp.itertuples():
                j = var_to_ind[str(row.itemid)]
                vals[j] = row.value
                mask[j] = 1.0
            if mask.sum() == 0:
                continue
            time_norm = float(t) / max_minutes
            rows.append(np.concatenate([vals, mask, [time_norm]]))
        if not rows:
            continue
        tensors.append(np.stack(rows, axis=0))
        labels.append(label_map[int(hadm_id)])

    if not tensors:
        raise RuntimeError("No sequences built; check data paths and fold ids")

    max_len = max(s.shape[0] for s in tensors)
    out = np.zeros((len(tensors), max_len, 2 * d + 1), dtype=np.float32)
    for i, seq in enumerate(tensors):
        out[i, : seq.shape[0]] = seq
    return out, np.array(labels, dtype=np.int64)


def _pad_seq_len(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Pad time axis so train/val/test can be stacked for pretrain."""
    if arr.shape[1] >= seq_len:
        return arr[:, :seq_len]
    out = np.zeros((arr.shape[0], seq_len, arr.shape[2]), dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    return out


def build_fold_tensors(
    cohort: pd.DataFrame,
    data: pd.DataFrame,
    features: List[str],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    max_obs: int,
    days_before_discharge: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    max_minutes = float((days_before_discharge + 1) * 24 * 60)
    data, ids = ids_in_data(
        data,
        {"train": train_ids, "val": val_ids, "test": test_ids, "sup_ids": None},
        logger=None,
    )
    ids["sup_ids"] = np.concatenate([ids["train"], ids["val"], ids["test"]])
    data = remove_features_not_in_train(data, ids["train"], logger=None)

    ts_map = {h: i for i, h in enumerate(ids["sup_ids"])}
    data = data.assign(ts_ind=data.hadm_id.map(ts_map))
    train_ind = [ts_map[h] for h in ids["train"] if h in ts_map]
    means_stds = compute_means_stds_df(data, train_ind)
    label_map = cohort.set_index("hadm_id")["label"].astype(int).to_dict()

    def pack(hadm_ids):
        return _build_sequences(
            data,
            np.array(hadm_ids),
            features,
            means_stds,
            label_map,
            max_obs,
            max_minutes,
        )

    X_train, y_train = pack(ids["train"])
    X_val, y_val = pack(ids["val"])
    X_test, y_test = pack(ids["test"])

    max_len = max(X_train.shape[1], X_val.shape[1], X_test.shape[1])
    X_train = _pad_seq_len(X_train, max_len)
    X_val = _pad_seq_len(X_val, max_len)
    X_test = _pad_seq_len(X_test, max_len)

    X_unlab = np.concatenate([X_train, X_val, X_test], axis=0)
    X_pre, X_pre_val = train_test_split(X_unlab, test_size=0.2, random_state=42)

    return {
        "finetune": {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        },
        "pretrain": {"X_train": X_pre, "X_val": X_pre_val},
    }


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
    for split_name, arrays in packs.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in arrays.items():
            torch.save(torch.tensor(arr), split_dir / f"{name}.pt")

    n_feat = (packs["finetune"]["X_train"].shape[2] - 1) // 2
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
