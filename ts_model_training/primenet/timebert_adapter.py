"""Helpers for native TimeBERT (no third_party imports)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def primenet_seq_len_cap(args) -> int:
    """TimeBERT ``pos_emb`` size; must match between pretrain checkpoint and finetune."""
    mp = getattr(args, "model_params", {}) or {}
    return int(mp.get("max_obs", getattr(args, "max_obs", 512)))


def infer_max_len_from_bert_checkpoint(ckpt_path) -> Optional[int]:
    """Read ``pos_emb`` rows from a saved TimeBERT state dict."""
    path = Path(ckpt_path)
    if not path.is_file():
        return None
    ckpt = _torch_load(str(path))
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    if not isinstance(state, dict):
        return None
    weight = state.get("pos_emb.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def _means_stds_to_dataframe(means_stds: Any):
    """Rebuild means/stds as DataFrame with index=itemid, columns mean/std."""
    import pandas as pd

    def _key(x):
        s = str(x)
        return int(x) if s.lstrip("-").isdigit() else x

    if isinstance(means_stds, dict):
        rows = {
            _key(itemid): {"mean": float(stats["mean"]), "std": float(stats["std"])}
            for itemid, stats in means_stds.items()
        }
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "itemid"
        return df[["mean", "std"]]

    if isinstance(means_stds, pd.DataFrame):
        df = means_stds.copy()
        if "itemid" in df.columns:
            df = df.set_index("itemid")
        df.index = [_key(x) for x in df.index]
        df.index.name = "itemid"
        df["mean"] = df["mean"].astype(float)
        df["std"] = df["std"].astype(float)
        return df[["mean", "std"]]

    raise TypeError(f"Unsupported means_stds type: {type(means_stds)}")


def dump_primenet_saved_variables(
    path,
    pt_variables: Any,
    pt_means_stds: Any,
    input_dim: int,
    max_len: int,
) -> None:
    """Write stats in a pandas-version-portable format (dict + plain lists)."""
    import pandas as pd

    variables = list(pt_variables)
    variables = [
        int(v) if not isinstance(v, str) or str(v).lstrip("-").isdigit() else v
        for v in variables
    ]

    if isinstance(pt_means_stds, dict):
        means_dict = {
            int(k) if str(k).lstrip("-").isdigit() else k: {
                "mean": float(v["mean"]),
                "std": float(v["std"]),
            }
            for k, v in pt_means_stds.items()
        }
    else:
        df = pt_means_stds
        if isinstance(df, pd.DataFrame):
            if "itemid" in df.columns:
                it = df["itemid"]
                means = df["mean"]
                stds = df["std"]
            else:
                it = df.index
                means = df["mean"]
                stds = df["std"]
            means_dict = {}
            for itemid, mean, std in zip(it, means, stds):
                key = int(itemid) if str(itemid).lstrip("-").isdigit() else itemid
                means_dict[key] = {"mean": float(mean), "std": float(std)}
        else:
            raise TypeError(f"Unsupported means_stds type for dump: {type(pt_means_stds)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "format": "primenet_stats_v2",
                "variables": variables,
                "means_stds": means_dict,
                "input_dim": int(input_dim),
                "max_len": int(max_len),
            },
            f,
            protocol=4,
        )


def load_primenet_saved_variables(path) -> Tuple[Any, Any, int, Optional[int]]:
    """Load ``primenet_saved_variables.pkl`` (legacy tuple or portable v2 dict)."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
    except TypeError as e:
        # Classic symptom: HPC pandas>=2.3 StringDtype vs Colab pandas<2.3
        raise RuntimeError(
            f"Failed to unpickle {path} ({e}). "
            "Usually a pandas version mismatch (need pandas>=2.3.0 to read HPC pickles). "
            "Fix: pip install -U 'pandas>=2.3.0' then re-run finetune "
            "(or re-export primenet_saved_variables with dump_primenet_saved_variables)."
        ) from e

    if isinstance(blob, dict) and blob.get("format") == "primenet_stats_v2":
        variables = list(blob["variables"])
        means_stds = _means_stds_to_dataframe(blob["means_stds"])
        return variables, means_stds, int(blob["input_dim"]), int(blob["max_len"])

    if not isinstance(blob, (tuple, list)):
        raise ValueError(f"Unexpected primenet_saved_variables type: {type(blob)}")

    if len(blob) == 3:
        variables, means_stds, input_dim = blob
        max_len = None
    elif len(blob) >= 4:
        variables, means_stds, input_dim, max_len = blob[0], blob[1], blob[2], blob[3]
    else:
        raise ValueError(f"Unexpected primenet_saved_variables format ({len(blob)} items)")

    variables = list(variables)
    means_stds = _means_stds_to_dataframe(means_stds)
    return variables, means_stds, int(input_dim), (None if max_len is None else int(max_len))


def resolve_primenet_max_len(
    args,
    saved_max_len: Optional[int] = None,
    ckpt_path=None,
) -> int:
    """Pick TimeBERT max_length for model construction (not collator padding)."""
    if saved_max_len is not None:
        return int(saved_max_len)
    path = ckpt_path or getattr(args, "pt_dict_path", None)
    inferred = infer_max_len_from_bert_checkpoint(path) if path else None
    if inferred is not None:
        return inferred
    return primenet_seq_len_cap(args)


def apply_primenet_max_len_for_finetune(args) -> int:
    """Ensure finetune builds TimeBERT with the same max_length as the pretrain ckpt."""
    saved = None
    pt_var = getattr(args, "pt_var_path", None)
    if pt_var and Path(pt_var).is_file():
        _, _, _, saved = load_primenet_saved_variables(pt_var)
    max_len = resolve_primenet_max_len(
        args, saved_max_len=saved, ckpt_path=getattr(args, "pt_dict_path", None)
    )
    args.primenet_max_len = max_len
    if hasattr(args, "logger"):
        args.logger.write(f"PrimeNet TimeBERT max_length={max_len}")
    return max_len


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


def classification_pooling(model, snapshot: torch.Tensor, dim: int) -> torch.Tensor:
    """TimeBERT pooled representation (before the classification MLP)."""
    observed_data, observed_mask, observed_tp = split_snapshot(snapshot, dim)
    x = torch.cat((observed_data, observed_mask), 2)
    outputs = model.bert(x, observed_tp)
    return outputs["cls_pooling"]


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
    ckpt_len = state.get("pos_emb.weight")
    model_len = model.bert.pos_emb.weight.shape[0]
    if ckpt_len is not None and int(ckpt_len.shape[0]) != model_len:
        raise RuntimeError(
            f"TimeBERT max_length mismatch: model pos_emb={model_len}, "
            f"checkpoint pos_emb={int(ckpt_len.shape[0])}. "
            "Re-run pretrain or set args.primenet_max_len to match the checkpoint "
            "before building the finetune model."
        )
    model.bert.load_state_dict(state)


def pretrain_forward(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, None]:
    value_batch = batch["value"]
    time_batch = batch["time"]
    mask_batch = batch["mask"]
    x_batch = torch.cat([value_batch, mask_batch], dim=-1)
    out = model(x_batch, time_batch)
    return out["loss"], None


def eval_pretrain_epoch(model, dataloader, device) -> dict: # float to dict since we have return loss + val_acc
    return eval_pretrain_loader(model, dataloader, device)


# Re-export for batcher imports
__all__ = [
    "apply_primenet_max_len_for_finetune",
    "build_pretrain_dataloaders",
    "build_pretrain_model",
    "build_classification_model",
    "classification_forward",
    "classification_pooling",
    "dump_primenet_saved_variables",
    "infer_max_len_from_bert_checkpoint",
    "load_bert_checkpoint",
    "load_primenet_saved_variables",
    "pretrain_forward",
    "eval_pretrain_epoch",
    "primenet_params",
    "primenet_seq_len_cap",
    "resolve_primenet_max_len",
]
