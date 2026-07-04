"""
PrimeNet / TimeBERT time-series model (native implementation).

Parallel to STRATS_TS in ts_strats.py: registered in factory.py and trained
via the unified Preprocessor → Batcher → Trainer pipeline.

Finetune fuses age/gender demographics the same way as other ChemoTree models
(demo_emb + concat with TimeBERT pooling before the classification head).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_model_training.primenet.timebert_adapter import (
    build_classification_model,
    build_pretrain_model,
    classification_pooling,
    pretrain_forward,
    primenet_params,
    primenet_seq_len_cap,
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
        max_len = int(getattr(args, "primenet_max_len", None) or primenet_seq_len_cap(args))

        if self.pretrain:
            self.core = build_pretrain_model(args, max_len, self.dim)
        else:
            self.core = build_classification_model(args, self.dim, max_length=max_len)
            params = primenet_params(args)
            ts_h = params["rec_hidden"]
            demo_h = int(getattr(args, "hid_dim_demo", 64))
            demo_dim = int(getattr(args, "D", 0))
            if demo_dim <= 0:
                raise ValueError("args.D (demographics dim) must be set before building PRIMENET_TS")

            self.demo_emb = nn.Sequential(
                nn.Linear(demo_dim, demo_h * 2),
                nn.Tanh(),
                nn.Linear(demo_h * 2, demo_h),
            )
            self.classification_head = nn.Sequential(
                nn.Linear(ts_h + demo_h, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, 2),
            )

    @property
    def bert(self):
        return self.core.bert

    def forward(
        self,
        snapshot=None,
        labels=None,
        demo=None,
        value=None,
        time=None,
        mask=None,
        **kwargs,
    ):
        if self.pretrain:
            batch = {"value": value, "time": time, "mask": mask}
            return pretrain_forward(self.core, batch)

        if demo is None:
            raise ValueError("PrimeNet finetune requires demo (age/gender) in the batch")

        ts_emb = classification_pooling(self.core, snapshot, self.dim)
        demo_emb = self.demo_emb(demo)
        fused = torch.cat((ts_emb, demo_emb), dim=-1)
        logits = self.classification_head(fused)
        return logits, None

    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels.long().view(-1))

    def predict(self, snapshot=None, demo=None, **kwargs):
        logits, _ = self.forward(snapshot=snapshot, demo=demo, **kwargs)
        return torch.softmax(logits, dim=-1)[:, 1]

    def config_summary(self) -> dict:
        """Training hyperparameters (analogous to STRATS hid_dim / num_layers)."""
        return primenet_params(self.args)
