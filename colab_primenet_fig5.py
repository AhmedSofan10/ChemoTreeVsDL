#!/usr/bin/env python3
"""Colab entry: STraTS Fig.5 PrimeNet pipeline (phased runs)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PrimeNet Fig.5 on Colab (wraps scripts/run_primenet_fig5.py)"
    )
    parser.add_argument(
        "--phase",
        default="pretrain-cohort",
        choices=[
            "prepare",
            "pretrain-cohort",
            "pretrain-mimicall",
            "finetune",
            "collect",
            "all",
        ],
    )
    parser.add_argument("--scenarios", default="nf")
    parser.add_argument("--mimic-all-cohort", default="mimic_all_10pct")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-extract-mimic-all", action="store_true")
    parser.add_argument("--skip-subset-mimic-all", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    args, extra = parser.parse_known_args()

    cmd = [
        sys.executable,
        "scripts/run_primenet_fig5.py",
        "--phase",
        args.phase,
        "--scenarios",
        args.scenarios,
        "--mimic-all-cohort",
        args.mimic_all_cohort,
    ]
    if args.skip_prepare:
        cmd.append("--skip-prepare")
    if args.skip_extract_mimic_all:
        cmd.append("--skip-extract-mimic-all")
    if args.skip_subset_mimic_all:
        cmd.append("--skip-subset-mimic-all")
    if args.fast:
        cmd.append("--fast")
    if args.no_collect:
        cmd.append("--no-collect")
    cmd.extend(extra)

    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
