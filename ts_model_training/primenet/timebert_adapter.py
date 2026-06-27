"""Helpers for native TimeBERT (no third_party imports)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ts_model_training.primenet.timebert import (
    TimeBERTConfig,
    TimeBERTForClassification,
    TimeBERTForPretrainingV2,
    build_pretrain_dataloaders,
    eval_pretrain_loader,
)


def _torch_load(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def device_from_args(args) -> torch.device:
    if hasattr(args, "device") and args.device is not None:
        return args.device if isinstance(args.device, torch.device) else torch.device(args.device)
    dev = str(getattr(args, "dev", "0"))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dev}")
    return torch.device("cpu")


def primenet_params(args) -> Dict[str, Any]:
    mp = getattr(args, "model_params", {}) or {}
    return {
        "pretrain_tasks": mp.get("pretrain_tasks", getattr(args, "pretrain_tasks", "full2")),
        "rec_hidden": int(mp.get("rec_hidden", getattr(args, "rec_hidden", 128))),
        "embed_time": int(mp.get("embed_time", getattr(args, "embed_time", 128))),
        "num_heads": int(mp.get("num_heads", getattr(args, "num_heads", 1))),
        "pretrain_pooling": mp.get("pretrain_pooling", getattr(args, "pretrain_pooling", "bert")),
        "finetune_pooling": mp.get("finetune_pooling", getattr(args, "finetune_pooling", "ave")),
        "lr": float(mp.get("lr", getattr(args, "lr", 1e-4))),
    }


def make_timebert_config(args, input_dim: int, max_len: int, pooling: str) -> TimeBERTConfig:
    params = primenet_params(args)
    return TimeBERTConfig(
        dataset="MIMIC-III",
        input_dim=input_dim,
        pretrain_tasks=params["pretrain_tasks"],
        cls_query=torch.linspace(0, 1.0, 128),
        hidden_size=params["rec_hidden"],
        embed_time=params["embed_time"],
        num_heads=params["num_heads"],
        learn_emb=True,
        freq=10.0,
        pooling=pooling,
        classify_pertp=False,
        max_length=max_len,
        dropout=0.3,
        temp=0.05,
    )


def split_snapshot(snapshot: torch.Tensor, dim: int):
    observed_data = snapshot[:, :, :dim]
    observed_mask = snapshot[:, :, dim : 2 * dim]
    observed_tp = snapshot[:, :, -1]
    return observed_data, observed_mask, observed_tp


def classification_forward(model, snapshot: torch.Tensor, dim: int) -> torch.Tensor:
    observed_data, observed_mask, observed_tp = split_snapshot(snapshot, dim)
    return model(torch.cat((observed_data, observed_mask), 2), observed_tp)


def build_pretrain_model(args, max_len: int, input_dim: int):
    params = primenet_params(args)
    config = make_timebert_config(args, input_dim, max_len, params["pretrain_pooling"])
    return TimeBERTForPretrainingV2(config).to(device_from_args(args))


def build_classification_model(args, input_dim: int, max_length: int = 512):
    params = primenet_params(args)
    pooling = (
        params["finetune_pooling"]
        if args.train_mode in ("finetune", "standard")
        else params["pretrain_pooling"]
    )
    config = make_timebert_config(args, input_dim, max_length, pooling)
    return TimeBERTForClassification(config).to(device_from_args(args))


def load_bert_checkpoint(model, ckpt_path: Path) -> None:
    ckpt = _torch_load(str(ckpt_path))
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.bert.load_state_dict(state)


def pretrain_forward(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, None]:
    value_batch = batch["value"]
    time_batch = batch["time"]
    mask_batch = batch["mask"]
    x_batch = torch.cat([value_batch, mask_batch], dim=-1)
    out = model(x_batch, time_batch)
    return out["loss"], None


def eval_pretrain_epoch(model, dataloader, device) -> float:
    return eval_pretrain_loader(model, dataloader, device)


# Re-export for batcher imports
__all__ = [
    "build_pretrain_dataloaders",
    "build_pretrain_model",
    "build_classification_model",
    "classification_forward",
    "load_bert_checkpoint",
    "pretrain_forward",
    "eval_pretrain_epoch",
    "primenet_params",
]
