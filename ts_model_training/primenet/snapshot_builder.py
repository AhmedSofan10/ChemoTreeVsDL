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
        return build_sequences(
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
        "meta": {
            "features": features,
            "means_stds": means_stds,
            "input_dim": len(features),
            "ts_map": ts_map,
            "train_hadm_ids": np.array(ids["train"]),
            "val_hadm_ids": np.array(ids["val"]),
            "test_hadm_ids": np.array(ids["test"]),
        },
    }
