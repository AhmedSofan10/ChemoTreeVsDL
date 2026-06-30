#!/usr/bin/env python3
"""
Run all STraTS Fig.5-style PrimeNet scenarios on MIMIC-IV NF (5 outer folds).

Scenarios (paper column names):
  cohort/all      — SSL pretrain on NF cohort once → full finetune (5 folds)
  cohort/final    — same pretrain → finetune with frozen TimeBERT backbone
  mimic_all/all   — SSL pretrain on all MIMIC admissions → full finetune on NF
  mimic_all/final — same → frozen backbone finetune
  none/none       — supervised finetune only (no SSL pretrain)
  none/final      — supervised finetune, frozen backbone (random init)

Examples:
  # Full matrix (needs MIMIC_IV/saved_data/ + raw MIMIC for mimic_all)
  python scripts/run_primenet_fig5.py --skip-prepare

  # Smoke test
  python scripts/run_primenet_fig5.py --fast --scenarios none_none

  # Only cohort columns + collect table
  python scripts/run_primenet_fig5.py --skip-prepare --scenarios cohort_all,cohort_final --collect
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.constants import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "ts_config_params.yaml"
DATASET = "MIMIC_IV"
NF_COHORT = "mimic_cohort_NF_30_days"
MIMIC_ALL_COHORT = "mimic_all"
NUM_FOLDS = 5

# Shared SSL checkpoints (one per pretrain source)
PT_PREFIX_COHORT = "fig5_pt_cohort"
PT_PREFIX_MIMIC_ALL = "fig5_pt_mimicall"

# Finetune result prefixes (one per Fig.5 column)
PREFIX_COHORT_ALL = "fig5_cohort_all"
PREFIX_COHORT_FINAL = "fig5_cohort_final"
PREFIX_MIMICALL_ALL = "fig5_mimicall_all"
PREFIX_MIMICALL_FINAL = "fig5_mimicall_final"
PREFIX_NONE_NONE = "fig5_none_none"
PREFIX_NONE_FINAL = "fig5_none_final"


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    prefix: str
    pretrain_source: str | None  # "cohort" | "mimic_all" | None
    freeze: bool = False
    supervised_only: bool = False


SCENARIOS: dict[str, Scenario] = {
    "cohort_all": Scenario(
        "cohort_all", "cohort/all", PREFIX_COHORT_ALL, "cohort", freeze=False
    ),
    "cohort_final": Scenario(
        "cohort_final", "cohort/final", PREFIX_COHORT_FINAL, "cohort", freeze=True
    ),
    "mimicall_all": Scenario(
        "mimicall_all", "mimic_all/all", PREFIX_MIMICALL_ALL, "mimic_all", freeze=False
    ),
    "mimicall_final": Scenario(
        "mimicall_final", "mimic_all/final", PREFIX_MIMICALL_FINAL, "mimic_all", freeze=True
    ),
    "none_none": Scenario(
        "none_none", "none/none", PREFIX_NONE_NONE, None, supervised_only=True
    ),
    "none_final": Scenario(
        "none_final", "none/final", PREFIX_NONE_FINAL, None, freeze=True, supervised_only=True
    ),
}


def _env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    root = str(PROJECT_ROOT)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.check_call(cmd, env=_env(), cwd=PROJECT_ROOT)


def _pretrain_ckpt_dir(cohort: str, prefix: str) -> Path:
    return (
        PROJECT_ROOT
        / DATASET
        / "saved_data"
        / "results"
        / cohort
        / "time_series"
        / "pretrain"
        / "primenet"
        / prefix
        / "fold_0"
        / "grid_none"
    )


def _ckpt_ready(path: Path) -> bool:
    return (path / "checkpoint_best.bin").is_file() and (
        path / "primenet_saved_variables.pkl"
    ).is_file()


def _main_base(fast: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "ts_model_training.main",
        "--dataset",
        DATASET,
        "--model_type",
        "primenet",
        "--grid",
        "none",
        "--feature_threshold",
        "--static_threshold",
        "0",
        "--hid_dim_demo",
        "64",
        "--config_path",
        str(CONFIG_PATH),
    ]
    if fast:
        cmd.append("--fast")
    return cmd


def run_pretrain(cohort: str, prefix: str, fast: bool, dry_run: bool) -> Path:
    out = _pretrain_ckpt_dir(cohort, prefix)
    if _ckpt_ready(out):
        print(f"[skip] pretrain ready: {out}")
        return out
    cmd = _main_base(fast) + [
        "--cohort",
        cohort,
        "--fold",
        "0",
        "--pretrain",
        "--prefix",
        prefix,
    ]
    _run(cmd, dry_run=dry_run)
    return out


def run_finetune_scenario(scenario: Scenario, ckpt_dir: Path | None, fast: bool, dry_run: bool) -> None:
    print(f"\n{'=' * 60}\nScenario: {scenario.label} ({scenario.key})\n{'=' * 60}")
    for fold in range(NUM_FOLDS):
        cmd = _main_base(fast) + [
            "--cohort",
            NF_COHORT,
            "--fold",
            str(fold),
            "--prefix",
            scenario.prefix,
        ]
        if ckpt_dir is not None:
            cmd += ["--load_ckpt_path", str(ckpt_dir)]
        if scenario.freeze:
            cmd.append("--freeze")
        if scenario.supervised_only:
            cmd.append("--supervised-only")
        _run(cmd, dry_run=dry_run)


def prepare_nf_cohort(dry_run: bool) -> None:
    _run(
        [sys.executable, "scripts/prepare_mimic_from_raw.py", "--cohort", NF_COHORT],
        dry_run=dry_run,
    )


def extract_mimic_all(dry_run: bool) -> None:
    to_ts = (
        PROJECT_ROOT
        / DATASET
        / "saved_data"
        / "processed_admission_features_for_ts"
        / MIMIC_ALL_COHORT
        / f"{MIMIC_ALL_COHORT}_admissions_labs_14_days_to_ts.csv.gz"
    )
    if to_ts.is_file():
        print(f"[skip] mimic_all labs already extracted: {to_ts}")
        return

    script = """
from ts_model_training.extractor import ExtractorPretrain
from ts_model_training.logger import Logger

class Args:
    dataset = "MIMIC_IV"
    cohort = "mimic_all"
    days_before_discharge = 14
    feature_threshold = False
    logger = Logger(None)

ExtractorPretrain(Args())
"""
    _run([sys.executable, "-c", script], dry_run=dry_run)


def collect_results(scenarios: list[Scenario], dry_run: bool) -> None:
    cmd = [sys.executable, "scripts/collect_primenet_results.py", "--train-mode", "finetune", "--fig5-table"]
    for s in scenarios:
        cmd += ["--prefix", s.prefix, "--label", s.label]
    _run(cmd, dry_run=dry_run)


def parse_scenarios(raw: str) -> list[Scenario]:
    if raw.strip().lower() == "all":
        return list(SCENARIOS.values())
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {unknown}. Choose from: {list(SCENARIOS)}")
    return [SCENARIOS[k] for k in keys]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PrimeNet Fig.5 scenario matrix (MIMIC-IV NF)")
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated keys or 'all'. Keys: " + ", ".join(SCENARIOS),
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Skip NF cohort prepare_mimic_from_raw")
    parser.add_argument(
        "--skip-extract-mimic-all",
        action="store_true",
        help="Skip mimic_all lab extraction (requires pre-built saved_data)",
    )
    parser.add_argument("--fast", action="store_true", help="Short training (--fast on ts_model_training.main)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--no-collect", action="store_true", help="Skip collect_primenet_results at the end")
    args = parser.parse_args()

    selected = parse_scenarios(args.scenarios)
    need_cohort_pt = any(s.pretrain_source == "cohort" for s in selected)
    need_mimicall_pt = any(s.pretrain_source == "mimic_all" for s in selected)

    if not args.skip_prepare:
        prepare_nf_cohort(args.dry_run)

    if need_mimicall_pt and not args.skip_extract_mimic_all:
        extract_mimic_all(args.dry_run)

    cohort_ckpt: Path | None = None
    mimicall_ckpt: Path | None = None

    if need_cohort_pt:
        cohort_ckpt = run_pretrain(NF_COHORT, PT_PREFIX_COHORT, args.fast, args.dry_run)

    if need_mimicall_pt:
        mimicall_ckpt = run_pretrain(MIMIC_ALL_COHORT, PT_PREFIX_MIMICALL, args.fast, args.dry_run)

    for scenario in selected:
        ckpt = None
        if scenario.pretrain_source == "cohort":
            ckpt = cohort_ckpt
        elif scenario.pretrain_source == "mimic_all":
            ckpt = mimicall_ckpt
        run_finetune_scenario(scenario, ckpt, args.fast, args.dry_run)

    if not args.no_collect and not args.dry_run:
        collect_results(selected, dry_run=False)

    print("\nDone. Finetune results under:")
    print(f"  {PROJECT_ROOT / DATASET / 'saved_data' / 'results' / NF_COHORT / 'time_series' / 'finetune' / 'primenet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
