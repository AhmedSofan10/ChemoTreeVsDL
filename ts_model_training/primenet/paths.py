"""PrimeNet data paths (tensor export cache)."""

from __future__ import annotations

from pathlib import Path

from config.constants import PRIMENET_DATA_DIR, PROJECT_ROOT


def primenet_tensor_dir(cohort: str, fold: int) -> Path:
    return PRIMENET_DATA_DIR / cohort / f"fold_{fold}"
