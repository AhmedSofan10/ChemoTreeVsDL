"""Pretrain data helpers for native TimeBERT."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ts_model_training.primenet.timebert.collator import CLDataCollator


class TimeDataset(Dataset):
    """Irregular time-series instances as (values, times, mask) tuples."""

    def __init__(self, data: List):
        super().__init__()
        self.data = []
        for instance in data:
            values, times, mask = instance
            if len(values) == len(times) == len(mask) and len(values) >= 2:
                self.data.append(instance)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def generate_irregular_samples(
    data: np.ndarray, input_dim: int
) -> Tuple[List, int]:
    """Convert padded snapshot arrays to irregular (values, times, mask) lists."""
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data)
    combined_data = []
    max_len = 0
    for i in range(data.shape[0]):
        zero_time_indices = torch.where(data[i, :, -1][1:] == 0)[0]
        curr_len = (
            zero_time_indices[0].item() + 1
            if len(zero_time_indices)
            else data.shape[1]
        )
        max_len = max(max_len, curr_len)
        values = data[i, :curr_len, :input_dim]
        times = data[i, :curr_len, -1]
        mask = data[i, :curr_len, input_dim : 2 * input_dim]
        combined_data.append([values, times, mask])
    return combined_data, max_len


def _collator_args_from_training_args(args) -> Namespace:
    mp = getattr(args, "model_params", {}) or {}
    return Namespace(
        pretrain_tasks=mp.get(
            "pretrain_tasks", getattr(args, "pretrain_tasks", "full2")
        ),
        batch_size=int(mp.get("batch_size", getattr(args, "batch_size", 64))),
        segment_num=int(mp.get("segment_num", getattr(args, "segment_num", 3))),
        mask_ratio_per_seg=float(
            mp.get("mask_ratio_per_seg", getattr(args, "mask_ratio_per_seg", 0.05))
        ),
        n=8000,
    )


def build_pretrain_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    args,
) -> Dict[str, Any]:
    """Build CLDataCollator dataloaders for TimeBERT pretraining."""
    max_pre = getattr(args, "max_pretrain_samples", None)
    if max_pre is None:
        mp = getattr(args, "model_params", {}) or {}
        max_pre = mp.get("max_pretrain_samples")
    if max_pre is not None:
        X_train = X_train[: int(max_pre)]
        X_val = X_val[: max(64, int(max_pre) // 5)]

    collator_args = _collator_args_from_training_args(args)
    input_dim = (X_train.shape[2] - 1) // 2
    X_train, train_max_len = generate_irregular_samples(X_train, input_dim)
    X_val, val_max_len = generate_irregular_samples(X_val, input_dim)
    max_len = max(train_max_len, val_max_len, 512)

    collator = CLDataCollator(max_len=max_len, args=collator_args)
    batch_size = min(
        min(len(X_val), collator_args.batch_size), collator_args.n
    )
    train_dataloader = DataLoader(
        TimeDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        TimeDataset(X_val),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    return {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": input_dim,
        "max_len": max_len,
        "collator_args": collator_args,
        "n_train_batches": len(train_dataloader),
        "n_val_batches": len(val_dataloader),
    }


def eval_pretrain_loader(model, dataloader, device) -> float:
    """Contrastive-learning accuracy on a pretrain validation loader."""
    model.eval()
    correct, total = 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            value_batch = batch["value"].to(device)
            time_batch = batch["time"].to(device)
            mask_batch = batch["mask"].to(device)
            x_batch = torch.cat([value_batch, mask_batch], dim=-1)
            out = model(x_batch, time_batch)
            correct += out["correct_num"]
            total += out["total_num"]
    return float(correct / total) if total else 0.0
