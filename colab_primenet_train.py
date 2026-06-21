#!/usr/bin/env python3
"""Colab entry: prepare data (optional) and train PrimeNet on MIMIC-IV NF."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config.constants import PRIMENET_VENDOR_ROOT, PROJECT_ROOT


def _env():
    env = os.environ.copy()
    root = str(PROJECT_ROOT)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["PRIMENET_ROOT"] = str(PRIMENET_VENDOR_ROOT.resolve())
    return env


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=_env())


def main():
    parser = argparse.ArgumentParser(description="ChemoTreeVsDL PrimeNet on Colab")
    parser.add_argument("--cohort", default="mimic_cohort_NF_30_days")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--prefix", default="colab_primenet")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if not (PRIMENET_VENDOR_ROOT / "timebert").is_dir():
        sys.exit("Missing vendored PrimeNet at third_party/PrimeNet")

    if not args.skip_prepare:
        _run([sys.executable, "scripts/prepare_mimic_from_raw.py", "--cohort", args.cohort])
    if args.prepare_only:
        return

    folds = range(5) if args.all_folds else [args.fold]
    template = [
        sys.executable,
        "-m",
        "ts_model_training.main",
        "--dataset",
        "MIMIC_IV",
        "--cohort",
        args.cohort,
        "--model_type",
        "primenet",
        "--feature_threshold",
        "--grid",
        "none",
        "--prefix",
        args.prefix,
        "--config_path",
        str(PROJECT_ROOT / "config" / "ts_config_params.yaml"),
    ]
    if args.fast:
        template.append("--fast")
    if args.skip_export:
        template.append("--skip-export")
    if args.skip_pretrain:
        template.append("--skip-pretrain")

    for fold in folds:
        _run(template + ["--fold", str(fold)])


if __name__ == "__main__":
    main()
