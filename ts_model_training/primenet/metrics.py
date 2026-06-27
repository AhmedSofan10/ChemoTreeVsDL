"""Classification metrics for PrimeNet (aligned with ts_model_training.evaluator)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, loss: float, thresh: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    pred_binary = (y_prob >= thresh).astype(int)
    return {
        "loss": float(loss),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(pr_auc),
        "precision": float(precision_score(y_true, pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true, pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true, pred_binary, zero_division=0)),
    }


@torch.no_grad()
def evaluate_primenet_classifier(model, dataloader, args, dim: int) -> dict:
    """Collect probs on a split and return STraTS-style metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    y_true, y_prob = [], []
    total_loss, n = 0.0, 0

    for batch, label in dataloader:
        batch = batch.to(args.device)
        label = label.to(args.device)
        observed_data = batch[:, :, :dim]
        observed_mask = batch[:, :, dim : 2 * dim]
        observed_tp = batch[:, :, -1]
        logits = model(torch.cat((observed_data, observed_mask), 2), observed_tp)
        loss = criterion(logits, label)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        total_loss += loss.item() * batch.size(0)
        n += batch.size(0)
        y_true.append(label.cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    avg_loss = total_loss / max(n, 1)
    return classification_metrics(y_true, y_prob, avg_loss)
