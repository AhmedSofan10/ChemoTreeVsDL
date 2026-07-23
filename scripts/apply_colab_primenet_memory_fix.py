#!/usr/bin/env python3
"""Idempotent T4 memory fixes for PrimeNet pretrain on Colab. Safe to run every session."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "chunked over features for T4 memory"
DATA_MARKER = "Pretrain val uses batch 1"


def patch_attention() -> bool:
    path = ROOT / "ts_model_training/primenet/timebert/models.py"
    text = path.read_text()
    if MARKER in text:
        return False
    old = '''    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \\
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.to(query.device).unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn.to(query.device)*value.unsqueeze(-3).to(query.device), -2), p_attn.to(query.device)'''
    new = '''    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute scaled dot-product attention (chunked over features for T4 memory)."
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        val = value.squeeze(1)
        feat_dim = val.size(-1)

        if mask is None:
            p_attn = F.softmax(scores, dim=-2)
            if dropout is not None:
                p_attn = dropout(p_attn)
            out = torch.einsum("bhql,bld->bhqd", p_attn, val)
            return out, p_attn

        m = mask.to(query.device)
        out = scores.new_zeros(scores.size(0), scores.size(1), scores.size(2), feat_dim)
        p_attn = None
        chunk = 32
        for d0 in range(0, feat_dim, chunk):
            d1 = min(d0 + chunk, feat_dim)
            scores_d = scores.unsqueeze(-1).expand(-1, -1, -1, -1, d1 - d0)
            scores_d = scores_d.masked_fill(m[..., d0:d1].unsqueeze(-3) == 0, -1e9)
            p_attn = F.softmax(scores_d, dim=-2)
            if dropout is not None:
                p_attn = dropout(p_attn)
            out[..., d0:d1] = torch.sum(
                p_attn * value[..., d0:d1].unsqueeze(-3),
                dim=-2,
            )
        return out, p_attn'''
    if old not in text:
        raise RuntimeError(f"Could not patch attention in {path} (unexpected content)")
    path.write_text(text.replace(old, new, 1))
    return True


def patch_dataloader() -> bool:
    path = ROOT / "ts_model_training/primenet/timebert/data.py"
    text = path.read_text()
    changed = False

    if "max(train_max_len, val_max_len, 512)" in text:
        text = text.replace(
            "max_len = max(train_max_len, val_max_len, 512)",
            "obs_cap = int((getattr(args, 'model_params', {}) or {}).get('max_obs', getattr(args, 'max_obs', 512)))\n"
            "    max_len = min(max(train_max_len, val_max_len, 1), obs_cap)",
        )
        changed = True

    if DATA_MARKER not in text and "def build_pretrain_dataloaders" in text:
        if "eval_batch_size" not in text:
            text = text.replace(
                "    batch_size = min(\n        min(len(X_val), collator_args.batch_size), collator_args.n\n    )",
                "    train_batch_size = min(len(X_train), collator_args.batch_size, collator_args.n)\n"
                "    eval_batch_size = 1  # Pretrain val uses batch 1",
            )
            text = text.replace(
                "batch_size=batch_size,\n        shuffle=True,",
                "batch_size=train_batch_size,\n        shuffle=True,",
            )
            text = text.replace(
                "batch_size=batch_size,\n        shuffle=False,",
                "batch_size=eval_batch_size,\n        shuffle=False,",
            )
            changed = True

    if changed:
        path.write_text(text)
    return changed


def patch_fast_batch_size() -> bool:
    path = ROOT / "ts_model_training/envmanager.py"
    text = path.read_text()
    if '"batch_size": 8,' in text or '"batch_size": 4,' in text:
        return False
    if '"batch_size": 16,' not in text:
        return False
    path.write_text(text.replace('"batch_size": 16,', '"batch_size": 8,', 1))
    return True


def main() -> None:
    patches = []
    if patch_attention():
        patches.append("timebert attention (chunked)")
    if patch_dataloader():
        patches.append("pretrain dataloader (max_len + eval batch 1)")
    if patch_fast_batch_size():
        patches.append("fast batch_size 8")
    if patches:
        print("Applied Colab memory fixes:", ", ".join(patches))
    else:
        print("Colab memory fixes already applied.")


if __name__ == "__main__":
    main()
