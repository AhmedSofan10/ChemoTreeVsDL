#!/usr/bin/env python3
"""Aggregate test metrics across outer CV folds (STraTS or PrimeNet)."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np

METRIC_KEYS = ("auroc", "auprc", "f1", "precision", "recall", "loss")


def parse_final_test_line(log_text: str) -> dict | None:
    for line in log_text.splitlines():
        if "Final test res:" not in line:
            continue
        payload = line.split("Final test res:", 1)[1].strip()
        payload = re.sub(r"np\.float64\(([^)]+)\)", r"\1", payload)
        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Summarize test metrics over folds")
    parser.add_argument("--cohort", default="mimic_cohort_NF_30_days")
    parser.add_argument("--prefix", default="colab_primenet")
    parser.add_argument("--grid", default="none")
    parser.add_argument("--model", default="primenet")
    parser.add_argument("--dataset", default="MIMIC_IV")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()

    if args.results_root:
        base = Path(args.results_root)
    else:
        base = (
            Path(args.dataset)
            / "saved_data"
            / "results"
            / args.cohort
            / "time_series"
            / "standard"
            / args.model
            / args.prefix
        )

    rows = []
    for fold in range(args.folds):
        log_path = base / f"fold_{fold}" / f"grid_{args.grid}" / "log.txt"
        if not log_path.is_file():
            print(f"fold {fold}: MISSING ({log_path})")
            continue
        metrics = parse_final_test_line(log_path.read_text())
        if not metrics:
            print(f"fold {fold}: no Final test res in log")
            continue
        rows.append({"fold": fold, **{k: metrics.get(k) for k in METRIC_KEYS}})

    if not rows:
        print("No completed folds found.")
        return 1

    print(f"\nResults: {base} / fold_*/grid_{args.grid}/\n")
    print(f"{'fold':>4}  {'auroc':>7}  {'auprc':>7}  {'f1':>7}  {'prec':>7}  {'recall':>7}")
    print("-" * 48)
    for r in rows:
        print(
            f"{r['fold']:>4}  {r['auroc']:>7.4f}  {r['auprc']:>7.4f}  {r['f1']:>7.4f}  "
            f"{r['precision']:>7.4f}  {r['recall']:>7.4f}"
        )

    arr = {k: np.array([r[k] for r in rows if r.get(k) is not None], dtype=float) for k in METRIC_KEYS}
    print("-" * 48)
    print(
        f"mean  {arr['auroc'].mean():>7.4f}  {arr['auprc'].mean():>7.4f}  {arr['f1'].mean():>7.4f}  "
        f"{arr['precision'].mean():>7.4f}  {arr['recall'].mean():>7.4f}"
    )
    print(
        f"std   {arr['auroc'].std(ddof=0):>7.4f}  {arr['auprc'].std(ddof=0):>7.4f}  {arr['f1'].std(ddof=0):>7.4f}  "
        f"{arr['precision'].std(ddof=0):>7.4f}  {arr['recall'].std(ddof=0):>7.4f}"
    )
    print(f"\nCompleted {len(rows)}/{args.folds} folds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
