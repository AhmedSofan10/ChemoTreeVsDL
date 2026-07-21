#!/usr/bin/env python3
"""
Run STraTS Fig.5-style PrimeNet scenarios on MIMIC-IV NF (5 outer folds).

Split into HPC-friendly phases (24h job limit):
  prepare          — NF cohort + optional mimic_all extraction / 10% subset
  pretrain-cohort  — SSL on NF cohort only (fold 0)
  pretrain-mimicall — SSL on mimic_all (or mimic_all_10pct for testing)
  finetune         — NF finetune scenarios (uses existing checkpoints)
  collect          — aggregate metrics table
  all              — legacy: run everything in one job (not recommended on HPC)

Scenarios:
  cohort_all, cohort_final     — pretrain on NF cohort
  mimicall_all, mimicall_final — pretrain on mimic_all (or --mimic-all-cohort)
  none_none, none_final        — supervised only

Examples:
  # Job 1: NF cohort pretrain (~30 min)
  python scripts/run_primenet_fig5.py --phase pretrain-cohort --skip-prepare

  # Job 2: 10% mimic_all pretrain (pipeline validation)
  python scripts/subset_mimic_all_cohort.py --frac 0.1
  python scripts/run_primenet_fig5.py --phase pretrain-mimicall --skip-prepare \\
      --mimic-all-cohort mimic_all_10pct

  # Job 3: NF finetune (no mimicall scenarios)
  python scripts/run_primenet_fig5.py --phase finetune --skip-prepare \\
      --scenarios nf

  # Job 4: mimicall finetune (after mimicall pretrain)
  python scripts/run_primenet_fig5.py --phase finetune --skip-prepare \\
      --scenarios mimicall --mimic-all-cohort mimic_all_10pct

  # Smoke test
  python scripts/run_primenet_fig5.py --fast --phase finetune --scenarios none_none
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
DEFAULT_MIMIC_ALL_COHORT = "mimic_all_10pct"
NUM_FOLDS = 5

PT_PREFIX_COHORT = "fig5_pt_cohort"
PT_PREFIX_MIMIC_ALL = "fig5_pt_mimicall"

PREFIX_COHORT_ALL = "fig5_cohort_all"
PREFIX_COHORT_FINAL = "fig5_cohort_final"
PREFIX_MIMICALL_ALL = "fig5_mimicall_all"
PREFIX_MIMICALL_FINAL = "fig5_mimicall_final"
PREFIX_NONE_NONE = "fig5_none_none"
PREFIX_NONE_FINAL = "fig5_none_final"

PHASES = (
    "prepare",
    "pretrain-cohort",
    "pretrain-mimicall",
    "finetune",
    "collect",
    "all",
)

SCENARIO_GROUPS: dict[str, list[str]] = {
    "nf": ["cohort_all", "cohort_final", "none_none", "none_final"],
    "mimicall": ["mimicall_all", "mimicall_final"],
    "all": list(
        [
            "cohort_all",
            "cohort_final",
            "mimicall_all",
            "mimicall_final",
            "none_none",
            "none_final",
        ]
    ),
}


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
        / "mimic_all"
        / "mimic_all_admissions_labs_14_days_to_ts.csv.gz"
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


def subset_mimic_all(target: str, frac: float, dry_run: bool) -> None:
    cohort_path = PROJECT_ROOT / DATASET / "saved_data" / "cohorts" / f"{target}.csv.gz"
    if cohort_path.is_file():
        print(f"[skip] subset cohort exists: {cohort_path}")
        return
    _run(
        [
            sys.executable,
            "scripts/subset_mimic_all_cohort.py",
            "--frac",
            str(frac),
            "--target",
            target,
        ],
        dry_run=dry_run,
    )


def collect_results(scenarios: list[Scenario], dry_run: bool) -> None:
    cmd = [sys.executable, "scripts/collect_primenet_results.py", "--train-mode", "finetune", "--fig5-table"]
    for s in scenarios:
        cmd += ["--prefix", s.prefix, "--label", s.label]
    _run(cmd, dry_run=dry_run)


def parse_scenarios(raw: str) -> list[Scenario]:
    key = raw.strip().lower()
    if key in SCENARIO_GROUPS:
        keys = SCENARIO_GROUPS[key]
    elif key == "all":
        keys = SCENARIO_GROUPS["all"]
    else:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SCENARIOS]
    if unknown:
        raise SystemExit(
            f"Unknown scenario(s): {unknown}. "
            f"Keys: {list(SCENARIOS)}; groups: {list(SCENARIO_GROUPS)}"
        )
    return [SCENARIOS[k] for k in keys]


def _resolve_ckpt(
    scenario: Scenario,
    cohort_ckpt: Path | None,
    mimicall_ckpt: Path | None,
    mimic_all_cohort: str,
    *,
    dry_run: bool = False,
) -> Path | None:
    if scenario.pretrain_source == "cohort":
        ckpt = cohort_ckpt or _pretrain_ckpt_dir(NF_COHORT, PT_PREFIX_COHORT)
    elif scenario.pretrain_source == "mimic_all":
        ckpt = mimicall_ckpt or _pretrain_ckpt_dir(mimic_all_cohort, PT_PREFIX_MIMIC_ALL)
    else:
        return None
    if dry_run:
        return ckpt
    if not _ckpt_ready(ckpt):
        raise SystemExit(
            f"Missing pretrain checkpoint for {scenario.key}: {ckpt}\n"
            f"Run the matching pretrain phase first."
        )
    return ckpt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PrimeNet Fig.5 scenario matrix (MIMIC-IV NF)")
    parser.add_argument(
        "--phase",
        default="all",
        choices=PHASES,
        help="Pipeline stage for HPC job splitting (default: all)",
    )
    parser.add_argument(
        "--scenarios",
        default="nf",
        help="Comma-separated keys, or group: nf | mimicall | all (default: nf)",
    )
    parser.add_argument(
        "--mimic-all-cohort",
        default=DEFAULT_MIMIC_ALL_COHORT,
        help=f"Cohort for mimic_all SSL pretrain (default: {DEFAULT_MIMIC_ALL_COHORT})",
    )
    parser.add_argument(
        "--mimic-all-frac",
        type=float,
        default=0.1,
        help="When building subset, fraction of admissions (default: 0.1)",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Skip NF cohort prepare_mimic_from_raw")
    parser.add_argument(
        "--skip-extract-mimic-all",
        action="store_true",
        help="Skip full mimic_all lab extraction",
    )
    parser.add_argument(
        "--skip-subset-mimic-all",
        action="store_true",
        help="Skip auto-building mimic_all subset (use if artifacts already exist)",
    )
    parser.add_argument("--fast", action="store_true", help="Short training (--fast on ts_model_training.main)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--no-collect", action="store_true", help="Skip collect_primenet_results at the end")
    args = parser.parse_args()

    phase = args.phase
    mimic_all_cohort = args.mimic_all_cohort

    if phase == "prepare":
        if not args.skip_prepare:
            prepare_nf_cohort(args.dry_run)
        if not args.skip_extract_mimic_all:
            extract_mimic_all(args.dry_run)
        if mimic_all_cohort != "mimic_all" and not args.skip_subset_mimic_all:
            subset_mimic_all(mimic_all_cohort, args.mimic_all_frac, args.dry_run)
        return 0

    if phase == "pretrain-cohort":
        if not args.skip_prepare:
            prepare_nf_cohort(args.dry_run)
        run_pretrain(NF_COHORT, PT_PREFIX_COHORT, args.fast, args.dry_run)
        return 0

    if phase == "pretrain-mimicall":
        if mimic_all_cohort == "mimic_all":
            if not args.skip_extract_mimic_all:
                extract_mimic_all(args.dry_run)
        else:
            if not args.skip_extract_mimic_all:
                extract_mimic_all(args.dry_run)
            if not args.skip_subset_mimic_all:
                subset_mimic_all(mimic_all_cohort, args.mimic_all_frac, args.dry_run)
        run_pretrain(mimic_all_cohort, PT_PREFIX_MIMIC_ALL, args.fast, args.dry_run)
        return 0

    selected = parse_scenarios(args.scenarios)
    need_cohort_pt = any(s.pretrain_source == "cohort" for s in selected)
    need_mimicall_pt = any(s.pretrain_source == "mimic_all" for s in selected)

    if phase == "collect":
        collect_results(selected, args.dry_run)
        return 0

    if phase == "finetune":
        cohort_ckpt = _pretrain_ckpt_dir(NF_COHORT, PT_PREFIX_COHORT) if need_cohort_pt else None
        mimicall_ckpt = (
            _pretrain_ckpt_dir(mimic_all_cohort, PT_PREFIX_MIMIC_ALL) if need_mimicall_pt else None
        )
        need_ckpt_scenarios = [s for s in selected if s.pretrain_source]
        if need_ckpt_scenarios and not args.dry_run:
            for s in need_ckpt_scenarios:
                src = s.pretrain_source
                ckpt_path = (
                    cohort_ckpt
                    if src == "cohort"
                    else mimicall_ckpt or _pretrain_ckpt_dir(mimic_all_cohort, PT_PREFIX_MIMIC_ALL)
                )
                if ckpt_path is None or not _ckpt_ready(ckpt_path):
                    print(
                        f"\nERROR: missing pretrain for {s.key}.\n"
                        f"  Expected: {ckpt_path}\n"
                        f"  Run: --phase pretrain-cohort  (or pretrain-mimicall)\n"
                        f"  Or test without checkpoint: --scenarios none_none\n",
                        file=sys.stderr,
                    )
                    return 1
        for scenario in selected:
            ckpt = _resolve_ckpt(scenario, cohort_ckpt, mimicall_ckpt, mimic_all_cohort, dry_run=args.dry_run)
            run_finetune_scenario(scenario, ckpt, args.fast, args.dry_run)
        if not args.no_collect and not args.dry_run:
            collect_results(selected, dry_run=False)
        print("\nDone. Finetune results under:")
        print(
            f"  {PROJECT_ROOT / DATASET / 'saved_data' / 'results' / NF_COHORT / 'time_series' / 'finetune' / 'primenet'}"
        )
        return 0

    # phase == "all" (monolithic; not recommended on HPC)
    if not args.skip_prepare:
        prepare_nf_cohort(args.dry_run)

    if need_mimicall_pt:
        if mimic_all_cohort == "mimic_all":
            if not args.skip_extract_mimic_all:
                extract_mimic_all(args.dry_run)
        else:
            if not args.skip_extract_mimic_all:
                extract_mimic_all(args.dry_run)
            if not args.skip_subset_mimic_all:
                subset_mimic_all(mimic_all_cohort, args.mimic_all_frac, args.dry_run)

    cohort_ckpt: Path | None = None
    mimicall_ckpt: Path | None = None

    if need_cohort_pt:
        cohort_ckpt = run_pretrain(NF_COHORT, PT_PREFIX_COHORT, args.fast, args.dry_run)

    if need_mimicall_pt:
        mimicall_ckpt = run_pretrain(mimic_all_cohort, PT_PREFIX_MIMIC_ALL, args.fast, args.dry_run)

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
    print(
        f"  {PROJECT_ROOT / DATASET / 'saved_data' / 'results' / NF_COHORT / 'time_series' / 'finetune' / 'primenet'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
