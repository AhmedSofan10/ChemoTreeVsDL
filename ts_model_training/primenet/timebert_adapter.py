"""Vendored TimeBERT helpers shared by ts_primenet and the legacy train_loop."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ts_model_training.primenet.paths import get_primenet_root


def _torch_load(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def ensure_primenet_imports():
    root = get_primenet_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _import_pretrain_module():
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]]
    try:
        import pretrain as pn_pretrain
        return pn_pretrain
    finally:
        sys.argv = saved_argv


def device_from_args(args) -> torch.device:
    dev = str(getattr(args, "dev", "0"))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dev}")
    return torch.device("cpu")


def primenet_config_from_args(args) -> Dict[str, Any]:
    mp = getattr(args, "model_params", {}) or {}
    return {
        "pretrain_tasks": mp.get("pretrain_tasks", getattr(args, "pretrain_tasks", "full2")),
        "pretrain_niters": int(mp.get("pretrain_niters", getattr(args, "pretrain_niters", 2000))),
        "finetune_niters": int(mp.get("finetune_niters", getattr(args, "finetune_niters", 2000))),
        "batch_size": int(mp.get("batch_size", getattr(args, "batch_size", 64))),
        "lr": float(mp.get("lr", getattr(args, "lr", 1e-4))),
        "rec_hidden": int(mp.get("rec_hidden", getattr(args, "rec_hidden", 128))),
        "embed_time": int(mp.get("embed_time", getattr(args, "embed_time", 128))),
        "num_heads": int(mp.get("num_heads", getattr(args, "num_heads", 1))),
        "pretrain_pooling": mp.get("pretrain_pooling", getattr(args, "pretrain_pooling", "bert")),
        "finetune_pooling": mp.get("finetune_pooling", getattr(args, "finetune_pooling", "ave")),
        "pooling": mp.get("pooling", "bert"),
        "patience": int(mp.get("patience", getattr(args, "patience", 20))),
        "finetune_patience": int(
            mp.get("finetune_patience", getattr(args, "finetune_patience", 20))
        ),
        "seed": int(mp.get("seed", getattr(args, "seed", 0))),
        "dev": str(mp.get("dev", getattr(args, "dev", "0"))),
        "segment_num": int(mp.get("segment_num", getattr(args, "segment_num", 3))),
        "mask_ratio_per_seg": float(
            mp.get("mask_ratio_per_seg", getattr(args, "mask_ratio_per_seg", 0.05))
        ),
        "max_pretrain_samples": mp.get("max_pretrain_samples", getattr(args, "max_pretrain_samples", None)),
        "max_finetune_samples": mp.get("max_finetune_samples", getattr(args, "max_finetune_samples", None)),
    }


def split_snapshot(snapshot: torch.Tensor, dim: int):
    observed_data = snapshot[:, :, :dim]
    observed_mask = snapshot[:, :, dim : 2 * dim]
    observed_tp = snapshot[:, :, -1]
    return observed_data, observed_mask, observed_tp


def classification_forward(model, snapshot: torch.Tensor, dim: int) -> torch.Tensor:
    observed_data, observed_mask, observed_tp = split_snapshot(snapshot, dim)
    return model(torch.cat((observed_data, observed_mask), 2), observed_tp)


def build_pretrain_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    args,
    pn_args: Optional[Namespace] = None,
) -> Dict[str, Any]:
    """Build CLDataCollator dataloaders for TimeBERT pretraining."""
    ensure_primenet_imports()
    import utils as pn_utils
    from collator import CLDataCollator

    params = primenet_config_from_args(args)
    max_pre = params.get("max_pretrain_samples")
    if max_pre is not None:
        X_train = X_train[: int(max_pre)]
        X_val = X_val[: max(64, int(max_pre) // 5)]

    if pn_args is None:
        pn_pretrain = _import_pretrain_module()
        pn_args = pn_pretrain.args
        pn_args.pretrain_tasks = params["pretrain_tasks"]
        pn_args.batch_size = params["batch_size"]
        pn_args.lr = params["lr"]
        pn_args.rec_hidden = params["rec_hidden"]
        pn_args.embed_time = params["embed_time"]
        pn_args.num_heads = params["num_heads"]
        pn_args.learn_emb = True
        pn_args.pooling = params["pretrain_pooling"]
        pn_args.patience = params["patience"]
        pn_args.seed = params["seed"]
        pn_args.segment_num = params["segment_num"]
        pn_args.mask_ratio_per_seg = params["mask_ratio_per_seg"]
        pn_args.n = 8000
        pn_args.device = device_from_args(args)

    input_dim = (X_train.shape[2] - 1) // 2
    X_train, train_max_len = pn_utils.generate_irregular_samples(X_train, input_dim)
    X_val, val_max_len = pn_utils.generate_irregular_samples(X_val, input_dim)
    max_len = max(train_max_len, val_max_len, 512)

    collator = CLDataCollator(max_len=max_len, args=pn_args)
    batch_size = min(min(len(X_val), pn_args.batch_size), pn_args.n)
    train_dataloader = DataLoader(
        pn_utils.TimeDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        pn_utils.TimeDataset(X_val),
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
        "pn_args": pn_args,
        "n_train_batches": len(train_dataloader),
        "n_val_batches": len(val_dataloader),
    }


def build_pretrain_model(args, max_len: int, input_dim: int):
    ensure_primenet_imports()
    from timebert import TimeBERTConfig, TimeBERTForPretrainingV2

    params = primenet_config_from_args(args)
    pn_args = Namespace(
        pretrain_tasks=params["pretrain_tasks"],
        rec_hidden=params["rec_hidden"],
        embed_time=params["embed_time"],
        num_heads=params["num_heads"],
        learn_emb=True,
        freq=10.0,
        pooling=params["pretrain_pooling"],
        device=device_from_args(args),
    )
    config = TimeBERTConfig(
        input_dim=input_dim,
        pretrain_tasks=params["pretrain_tasks"],
        cls_query=torch.linspace(0, 1.0, 128),
        hidden_size=params["rec_hidden"],
        embed_time=params["embed_time"],
        num_heads=params["num_heads"],
        learn_emb=True,
        freq=10.0,
        pooling=params["pretrain_pooling"],
        max_length=max_len,
        dropout=0.3,
        temp=0.05,
    )
    model = TimeBERTForPretrainingV2(config).to(pn_args.device)
    return model, pn_args


def build_classification_model(args, input_dim: int, max_length: int = 512):
    ensure_primenet_imports()
    from timebert import TimeBERTConfig, TimeBERTForClassification

    params = primenet_config_from_args(args)
    pooling = (
        params["finetune_pooling"]
        if args.train_mode in ("finetune", "standard")
        else params["pretrain_pooling"]
    )
    config = TimeBERTConfig(
        dataset="MIMIC-III",
        input_dim=input_dim,
        cls_query=torch.linspace(0, 1.0, 128),
        hidden_size=params["rec_hidden"],
        embed_time=params["embed_time"],
        num_heads=params["num_heads"],
        learn_emb=True,
        freq=10.0,
        pooling=pooling,
        classify_pertp=False,
        max_length=max_length,
        dropout=0.3,
        temp=0.05,
    )
    return TimeBERTForClassification(config).to(device_from_args(args))


def load_bert_checkpoint(model, ckpt_path: Path) -> None:
    ckpt = _torch_load(str(ckpt_path))
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.bert.load_state_dict(state)


def pretrain_forward(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, None]:
    value_batch = batch["value"]
    time_batch = batch["time"]
    mask_batch = batch["mask"]
    x_batch = torch.cat([value_batch, mask_batch], dim=-1)
    out = model(x_batch, time_batch)
    return out["loss"], None


def run_pretrain_epoch(model, dataloader, optimizer, pn_args) -> None:
    pn_pretrain = _import_pretrain_module()
    pn_pretrain.train(pn_args, model, dataloader, optimizer)


def eval_pretrain_epoch(model, dataloader, pn_args) -> float:
    pn_pretrain = _import_pretrain_module()
    _, _, val_acc = pn_pretrain.eval(pn_args, model, dataloader)
    return float(val_acc)


def make_pretrain_optimizer(model, args):
    params = primenet_config_from_args(args)
    return optim.Adam(model.parameters(), lr=params["lr"])
