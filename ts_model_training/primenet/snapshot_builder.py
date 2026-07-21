"""Build PrimeNet snapshot tensors [N, T, 2*D+1] from ChemoTree lab tables."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ts_model_training.utils import (
    compute_means_stds_df,
    ids_in_data,
    remove_features_not_in_train,
)


def build_sequences(
    data: pd.DataFrame,
    hadm_ids: np.ndarray,
    features: List[str],
    means_stds: pd.DataFrame,
    label_map: Dict[int, int],
    max_obs: int,
    max_minutes: float,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray]:
    hadm_ids = np.asarray(hadm_ids)
    d = len(features)
    if hadm_ids.size == 0:
        return np.zeros((0, 1, 2 * d + 1), dtype=np.float32), np.array([], dtype=np.int64)

    if logger is not None:
        logger.write(f"\nbuild_sequences: starting on {hadm_ids.size} admissions")

    var_to_ind = {v: i for i, v in enumerate(features)}
    tensors = []
    labels = []

    sub = data.loc[data.hadm_id.isin(hadm_ids)].copy()
    ms = means_stds.reset_index() if "itemid" not in means_stds.columns else means_stds
    sub = sub.merge(ms, on="itemid", how="left")
    sub["value"] = (sub["value"] - sub["mean"]) / sub["std"]
    sub = sub.groupby(["hadm_id", "minute", "itemid"]).value.mean().reset_index()

    # Group once up front instead of re-scanning 
    admission_groups = {k: g for k, g in sub.groupby("hadm_id")}
    if logger is not None:
        logger.write(f"build_sequences: grouped rows for {len(admission_groups)} admissions, building tensors...")

    progress_every = max(1, hadm_ids.size // 10)
    for i, hadm_id in enumerate(hadm_ids):
        if logger is not None and i > 0 and i % progress_every == 0:
            logger.write(f"build_sequences: {i}/{hadm_ids.size} admissions processed")
        grp = admission_groups.get(hadm_id)
        if grp is None or grp.empty:
            continue
        times = sorted(grp["minute"].unique())
        if len(times) > max_obs:
            rng = np.random.default_rng(int(hadm_id) % (2**31))
            times = sorted(rng.choice(times, size=max_obs, replace=False))
        minute_groups = {k: g for k, g in grp.groupby("minute")}
        rows = []
        for t in times:
            t_grp = minute_groups.get(t)
            if t_grp is None:
                continue
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
    if logger is not None:
        logger.write(f"build_sequences: done, built {out.shape}")
    return out, np.array(labels, dtype=np.int64)


def pad_seq_len(arr: np.ndarray, seq_len: int) -> np.ndarray:
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
    pretrain_only: bool = False,
    logger=None,
) -> Dict[str, Dict[str, np.ndarray]]:
    max_minutes = float((days_before_discharge + 1) * 24 * 60)
    data, ids = ids_in_data(
        data,
        {"train": train_ids, "val": val_ids, "test": test_ids, "sup_ids": None},
        logger=logger,
    )
    ids["sup_ids"] = np.concatenate([ids["train"], ids["val"], ids["test"]])
    data = remove_features_not_in_train(data, ids["train"], logger=logger)

    ts_map = {h: i for i, h in enumerate(ids["sup_ids"])}
    data = data.assign(ts_ind=data.hadm_id.map(ts_map))
    train_ind = [ts_map[h] for h in ids["train"] if h in ts_map]
    means_stds = compute_means_stds_df(data, train_ind)
    if "label" in cohort.columns:
        label_map = cohort.set_index("hadm_id")["label"].astype(int).to_dict()
    else:
        # Unlabeled cohorts (e.g. mimic_all SSL pretrain): labels unused for pretrain packs.
        label_map = {int(h): 0 for h in cohort["hadm_id"].unique()}

    def pack(hadm_ids):
        hadm_ids = np.asarray(hadm_ids)
        if hadm_ids.size == 0:
            d = len(features)
            return (
                np.zeros((0, 1, 2 * d + 1), dtype=np.float32),
                np.array([], dtype=np.int64),
            )
        return build_sequences(
            data,
            hadm_ids,
            features,
            means_stds,
            label_map,
            max_obs,
            max_minutes,
            logger=logger,
        )

    meta = {
        "features": features,
        "means_stds": means_stds,
        "input_dim": len(features),
        "ts_map": ts_map,
        "train_hadm_ids": np.array(ids["train"]),
        "val_hadm_ids": np.array(ids["val"]),
        "test_hadm_ids": np.array(ids["test"]),
    }

    if pretrain_only:
        # Split ids before building dense tensors, so the pooled "pretrain" arrays (the only
        # split SSL pretrain uses) are built once instead of per-split then concatenated,
        # which OOMed on the full mimic_all cohort.
        pre_train_ids, pre_val_ids = train_test_split(
            ids["sup_ids"], test_size=0.2, random_state=42
        )
        if logger is not None:
            logger.write(f"\nbuild_fold_tensors: building pretrain train split ({len(pre_train_ids)} admissions)")
        X_pre, _ = pack(pre_train_ids)
        if logger is not None:
            logger.write(f"build_fold_tensors: building pretrain val split ({len(pre_val_ids)} admissions)")
        X_pre_val, _ = pack(pre_val_ids)
        max_len = max(X_pre.shape[1], X_pre_val.shape[1])
        X_pre = pad_seq_len(X_pre, max_len)
        X_pre_val = pad_seq_len(X_pre_val, max_len)
        if logger is not None:
            logger.write(f"build_fold_tensors: pretrain packs done, train {X_pre.shape}, val {X_pre_val.shape}")
        return {
            "pretrain": {"X_train": X_pre, "X_val": X_pre_val},
            "meta": meta,
        }

    X_train, y_train = pack(ids["train"])
    X_val, y_val = pack(ids["val"])
    X_test, y_test = pack(ids["test"])

    max_len = max(X_train.shape[1], X_val.shape[1])
    if X_test.shape[0] > 0:
        max_len = max(max_len, X_test.shape[1])
    X_train = pad_seq_len(X_train, max_len)
    X_val = pad_seq_len(X_val, max_len)
    X_test = pad_seq_len(X_test, max_len)

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
        "meta": meta,
    }
