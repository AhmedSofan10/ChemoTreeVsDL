"""Resolve vendored PrimeNet (third_party/PrimeNet)."""

from __future__ import annotations

import os
from pathlib import Path

from config.constants import PRIMENET_VENDOR_ROOT, PROJECT_ROOT


def get_primenet_root() -> Path:
    env = os.environ.get("PRIMENET_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = PRIMENET_VENDOR_ROOT
    if not (root / "timebert").is_dir():
        raise FileNotFoundError(
            f"Vendored PrimeNet not found at {root}. "
            "Expected third_party/PrimeNet from the ChemoTreeVsDL repo."
        )
    return root


def primenet_tensor_dir(cohort: str, fold: int) -> Path:
    from config.constants import PRIMENET_DATA_DIR

    return PRIMENET_DATA_DIR / cohort / f"fold_{fold}"
