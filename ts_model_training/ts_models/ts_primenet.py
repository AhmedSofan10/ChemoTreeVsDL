"""
PrimeNet / TimeBERT time-series model (native implementation).

Parallel to STRATS_TS in ts_strats.py: registered in factory.py and trained
via the unified Preprocessor → Batcher → Trainer pipeline.

Architecture lives in ts_model_training.primenet.timebert (modules, models, collator).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_model_training.primenet.timebert_adapter import (
    build_classification_model,
    build_pretrain_model,
    classification_forward,
    pretrain_forward,
    primenet_params,
)


class PRIMENET_TS(nn.Module):
    """TimeBERT for ChemoTree MIMIC-IV NF (pretrain + finetune)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_mode = args.train_mode
        self.dim = int(getattr(args, "input_dim", 0))
        if self.dim <= 0:
            raise ValueError("args.input_dim must be set before building PRIMENET_TS")

        self.pretrain = self.train_mode == "pretrain"
        self.finetune = self.train_mode == "finetune"
        max_len = int(getattr(args, "primenet_max_len", 512))

        if self.pretrain:
            self.core = build_pretrain_model(args, max_len, self.dim)
        else:
            self.core = build_classification_model(args, self.dim, max_length=max_len)

    @property
    def bert(self):
        return self.core.bert

    def forward(
        self,
        snapshot=None,
        labels=None,
        value=None,
        time=None,
        mask=None,
        **kwargs,
    ):
        if self.pretrain:
            batch = {"value": value, "time": time, "mask": mask}
            return pretrain_forward(self.core, batch)

        logits = classification_forward(self.core, snapshot, self.dim)
        return logits, None

    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels.long().view(-1))

    def predict(self, snapshot=None, **kwargs):
        logits, _ = self.forward(snapshot=snapshot, **kwargs)
        return torch.softmax(logits, dim=-1)[:, 1]

    def config_summary(self) -> dict:
        """Training hyperparameters (analogous to STRATS hid_dim / num_layers)."""
        return primenet_params(self.args)
