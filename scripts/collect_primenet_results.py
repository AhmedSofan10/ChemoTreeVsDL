#!/usr/bin/env python3
"""
Collect PrimeNet test metrics from local (or Colab) GPU runs.

Reads per-fold results under:
  MIMIC_IV/saved_data/results/<cohort>/time_series/{finetune|standard}/primenet/<prefix>/fold_*/grid_none/

Prefers results_final.csv (split=test); falls back to log.txt "Final test res:".

Examples:
  # One run (matches train_primenet_local.sh --prefix local_full)
  python scripts/collect_primenet_results.py --prefix local_full

  # Paper-style row (AUROC, AUC-PRC, F1 means over 5 folds)
  python scripts/collect_primenet_results.py --prefix local_full --fig5-row

  # Several configs (Fig. 5 columns once you have multiple prefixes)
  python scripts/collect_primenet_results.py \\
    --prefix pn_cohort_all --prefix pn_cohort_final --fig5-table

  # Save CSV summary next to results
  python scripts/collect_primenet_results.py --prefix local_full --out-csv summary_local_full.csv
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# repo root on PYTHONPATH when run from ChemoTreeVsDL
from config.constants import PROJECT_ROOT

METRICS = ("auroc", "auprc", "f1")
EXTRA_METRICS = ("precision", "recall", "loss")
ALL_METRICS = METRICS + EXTRA_METRICS


def parse_log_final_test(log_text: str) -> dict | None:
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


def resolve_results_base(
    dataset: str,
    cohort: str,
    prefix: str,
    model: str = "primenet",
    train_mode: str | None = None,
) -> tuple[Path, str]:
    """Return (base_path, train_mode_used)."""
    root = PROJECT_ROOT / dataset / "saved_data" / "results" / cohort / "time_series"
    if train_mode:
        base = root / train_mode / model / prefix
        return base, train_mode

    for mode in ("finetune", "standard"):
        candidate = root / mode / model / prefix
        if candidate.is_dir() and any((candidate / f"fold_{i}").exists() for i in range(5)):
            return candidate, mode
    # default expected layout for current pipeline
    return root / "finetune" / model / prefix, "finetune"


def read_fold_metrics(fold_dir: Path, grid: str = "none") -> dict | None:
    run_dir = fold_dir / f"grid_{grid}"
    if not run_dir.is_dir():
        alt = sorted(fold_dir.glob("grid_*"))
        if not alt:
            return None
        run_dir = alt[0]

    csv_path = run_dir / "results_final.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        test_rows = df[df["split"].astype(str).str.lower() == "test"]
        if not test_rows.empty:
            row = test_rows.iloc[-1]
            out = {k: float(row[k]) for k in ALL_METRICS if k in row and pd.notna(row[k])}
            if out:
                return out

    log_path = run_dir / "log.txt"
    if log_path.is_file():
        parsed = parse_log_final_test(log_path.read_text(errors="replace"))
        if parsed:
            return {k: float(parsed[k]) for k in ALL_METRICS if k in parsed}

    return None


def collect_prefix(
    dataset: str,
    cohort: str,
    prefix: str,
    folds: int,
    grid: str,
    train_mode: str | None,
    model: str,
) -> tuple[Path, str, list[dict]]:
    base, mode = resolve_results_base(dataset, cohort, prefix, model, train_mode)
    rows: list[dict] = []

    for fold in range(folds):
        fold_dir = base / f"fold_{fold}"
        metrics = read_fold_metrics(fold_dir, grid=grid)
        if metrics is None:
            rows.append({"fold": fold, "status": "missing", **{m: np.nan for m in ALL_METRICS}})
        else:
            rows.append({"fold": fold, "status": "ok", **metrics})

    return base, mode, rows


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows if r.get("status") == "ok" and key in r and not np.isnan(r[key])]
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def print_fold_table(prefix: str, base: Path, mode: str, rows: list[dict]) -> int:
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{'=' * 60}")
    print(f"prefix: {prefix}")
    print(f"path:   {base}")
    print(f"mode:   {mode}  |  folds: {n_ok}/{len(rows)}")
    print(f"{'=' * 60}")
    print(f"{'fold':>4}  {'auroc':>8}  {'auprc':>8}  {'f1':>8}  {'prec':>8}  {'recall':>8}")
    print("-" * 52)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['fold']:>4}  MISSING")
            continue
        print(
            f"{r['fold']:>4}  {r['auroc']:>8.4f}  {r['auprc']:>8.4f}  {r['f1']:>8.4f}  "
            f"{r.get('precision', float('nan')):>8.4f}  {r.get('recall', float('nan')):>8.4f}"
        )
    print("-" * 52)
    for label, fn in [("mean", np.mean), ("std", lambda x: np.std(x, ddof=0))]:
        parts = []
        for m in METRICS:
            vals = [r[m] for r in rows if r["status"] == "ok"]
            parts.append(f"{fn(vals):>8.4f}" if vals else f"{'n/a':>8}")
        print(f"{label:>4}  {parts[0]}  {parts[1]}  {parts[2]}")
    return n_ok


def fig5_row_line(label: str, rows: list[dict]) -> str:
    cells = []
    for m in METRICS:
        mu, _ = mean_std(rows, m)
        cells.append(f"{mu:.3f}" if not np.isnan(mu) else "n/a")
    return f"{label:20}  AUROC {cells[0]}  |  AUC-PRC {cells[1]}  |  F1 {cells[2]}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect PrimeNet local/Colab GPU results (NF MIMIC-IV folds)"
    )
    parser.add_argument("--dataset", default="MIMIC_IV")
    parser.add_argument("--cohort", default="mimic_cohort_NF_30_days")
    parser.add_argument("--model", default="primenet")
    parser.add_argument("--grid", default="none")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--train-mode",
        default=None,
        help="Force results subfolder: finetune, standard, or pretrain (default: auto)",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Run prefix (repeat for multiple). Default: scan all under primenet/",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Fig.5-style column label per --prefix (same order)",
    )
    parser.add_argument(
        "--fig5-row",
        action="store_true",
        help="Print one paper-style row (NF MIMIC-IV) for a single prefix",
    )
    parser.add_argument(
        "--fig5-table",
        action="store_true",
        help="Print comparison table when multiple --prefix given",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Write long-format CSV (prefix, fold, metrics)",
    )
    parser.add_argument(
        "--list-prefixes",
        action="store_true",
        help="List available primenet prefixes under results/ and exit",
    )
    args = parser.parse_args()

    results_root = (
        PROJECT_ROOT
        / args.dataset
        / "saved_data"
        / "results"
        / args.cohort
        / "time_series"
    )

    if args.list_prefixes:
        found = set()
        for mode in ("finetune", "standard", "pretrain"):
            mode_dir = results_root / mode / args.model
            if mode_dir.is_dir():
                for p in mode_dir.iterdir():
                    if p.is_dir():
                        found.add((p.name, mode))
        if not found:
            print(f"No prefixes under {results_root}/*/primenet/")
            return 1
        print(f"Available prefixes ({args.cohort}):")
        for name, mode in sorted(found):
            print(f"  {name:30}  [{mode}]")
        return 0

    prefixes = args.prefix
    if not prefixes:
        # default: try common local/colab names
        for candidate in ("local_full", "colab_primenet_full", "colab_primenet", "local_smoke"):
            base, mode = resolve_results_base(
                args.dataset, args.cohort, candidate, args.model, args.train_mode
            )
            if base.is_dir() and any((base / f"fold_{i}").exists() for i in range(5)):
                prefixes = [candidate]
                print(f"Auto-selected prefix: {candidate} ({mode})")
                break
        if not prefixes:
            print(
                "No --prefix given and no default run found. "
                "Use --list-prefixes or --prefix YOUR_PREFIX",
                file=sys.stderr,
            )
            return 1

    labels = args.label or [p for p in prefixes]
    if len(labels) != len(prefixes):
        print("--label count must match --prefix count", file=sys.stderr)
        return 1

    all_summaries = []
    any_ok = False

    for prefix, label in zip(prefixes, labels):
        base, mode, rows = collect_prefix(
            args.dataset,
            args.cohort,
            prefix,
            args.folds,
            args.grid,
            args.train_mode,
            args.model,
        )
        n_ok = print_fold_table(prefix, base, mode, rows)
        any_ok = any_ok or n_ok > 0

        mu_auroc, sd_auroc = mean_std(rows, "auroc")
        mu_auprc, sd_auprc = mean_std(rows, "auprc")
        mu_f1, sd_f1 = mean_std(rows, "f1")

        all_summaries.append(
            {
                "label": label,
                "prefix": prefix,
                "train_mode": mode,
                "path": str(base),
                "folds_ok": n_ok,
                "folds_total": args.folds,
                "auroc_mean": mu_auroc,
                "auroc_std": sd_auroc,
                "auprc_mean": mu_auprc,
                "auprc_std": sd_auprc,
                "f1_mean": mu_f1,
                "f1_std": sd_f1,
            }
        )

        for r in rows:
            if r["status"] != "ok":
                continue
            all_summaries.append(
                {
                    "label": label,
                    "prefix": prefix,
                    "train_mode": mode,
                    "path": str(base),
                    "fold": r["fold"],
                    "auroc": r["auroc"],
                    "auprc": r["auprc"],
                    "f1": r["f1"],
                    "precision": r.get("precision"),
                    "recall": r.get("recall"),
                }
            )

        if args.fig5_row and len(prefixes) == 1:
            print(f"\nFig.5-style row (NF MIMIC-IV) — {label}")
            print(fig5_row_line("PrimeNet", rows))
            print("\nSTraTS reference (cohort/all):  AUROC 0.779  |  AUC-PRC 0.193  |  F1 0.177")

    if args.fig5_table and len(prefixes) > 1:
        print(f"\n{'=' * 60}")
        print("Fig.5-style comparison (mean over folds)")
        print(f"{'=' * 60}")
        print(f"{'config':22}  {'AUROC':>8}  {'AUC-PRC':>8}  {'F1':>8}")
        print("-" * 52)
        for prefix, label in zip(prefixes, labels):
            _, _, rows = collect_prefix(
                args.dataset,
                args.cohort,
                prefix,
                args.folds,
                args.grid,
                args.train_mode,
                args.model,
            )
            m = [mean_std(rows, k)[0] for k in METRICS]
            print(
                f"{label:22}  {m[0]:>8.3f}  {m[1]:>8.3f}  {m[2]:>8.3f}"
                if not any(np.isnan(x) for x in m)
                else f"{label:22}  {'n/a':>8}  {'n/a':>8}  {'n/a':>8}"
            )
        print("\nSTraTS NF (MIMIC-IV) reference rows in paper Fig.5:")
        print("  cohort/all   0.779  0.193  0.177")
        print("  none/none    0.783  0.199  0.137")

    if args.out_csv:
        out_path = Path(args.out_csv)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        # long format: per-fold rows only
        fold_rows = [s for s in all_summaries if "fold" in s]
        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(out_path, index=False)
            print(f"\nWrote {out_path}")

    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
