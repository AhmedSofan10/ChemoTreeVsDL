#!/usr/bin/env python3
"""
Generate ChemoTree/STraTS Fig.5-style heatmaps (matplotlib), matching the paper layout.

Paper colors cells by (value - none/none baseline), diverging RdBu, ±0.06.
We annotate mean ± std (paper showed means only).

Outputs (next to this script / --out-dir):
  fig_primenet_fig5.pdf /.png
  fig_strats_fig5.pdf /.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# ---- Verified 5-fold means / stds + paper none/none baselines ----

# Columns: all_adm/all, all_adm/final, cohort/all, cohort/final, none/none
COL_LABELS_PT = [("all adm", 0, 2), ("cohort", 2, 4), ("none", 4, 5)]
COL_LABELS_FT = ["all", "final", "all", "final", "none"]
PT_COLORS = ["#9ecae1", "#7fcdbb", "#bdbdbd"]
FT_COLORS = ["#a1d99b", "#ffe08a", "#a1d99b", "#ffe08a", "#bdbdbd"]

# Paper STraTS none/none baselines (Rahimi et al. Fig.5)
BASE = {
    "AUROC": {"Aplasia (MIMIC-IV)": 0.864, "NF (MIMIC-IV)": 0.783},
    "AUPRC": {"Aplasia (MIMIC-IV)": 0.622, "NF (MIMIC-IV)": 0.199},
    "F1 score": {"Aplasia (MIMIC-IV)": 0.545, "NF (MIMIC-IV)": 0.137},
}

# PrimeNet: rows = Aplasia, NF  (paper order: aplasia above NF)
PRIMENET = {
    "AUROC": {
        "rows": ["Aplasia (MIMIC-IV)", "NF (MIMIC-IV)"],
        "mean": np.array(
            [
                [0.842, 0.783, 0.854, 0.808, 0.864],
                [0.721, 0.664, 0.769, 0.678, 0.783],
            ]
        ),
        "std": np.array(
            [
                [0.031, 0.042, 0.027, 0.033, np.nan],
                [0.021, 0.041, 0.030, 0.022, np.nan],
            ]
        ),
    },
    "AUPRC": {
        "rows": ["Aplasia (MIMIC-IV)", "NF (MIMIC-IV)"],
        "mean": np.array(
            [
                [0.568, 0.497, 0.615, 0.514, 0.622],
                [0.135, 0.116, 0.168, 0.119, 0.199],
            ]
        ),
        "std": np.array(
            [
                [0.067, 0.074, 0.077, 0.073, np.nan],
                [0.053, 0.025, 0.047, 0.019, np.nan],
            ]
        ),
    },
    "F1 score": {
        "rows": ["Aplasia (MIMIC-IV)", "NF (MIMIC-IV)"],
        "mean": np.array(
            [
                [0.589, 0.418, 0.597, 0.464, 0.545],
                [0.190, 0.170, 0.210, 0.160, 0.137],
            ]
        ),
        "std": np.array(
            [
                [0.053, 0.057, 0.041, 0.052, np.nan],
                [0.034, 0.045, 0.039, 0.018, np.nan],
            ]
        ),
    },
}

# STraTS NF only (all adm not re-run)
STRATS = {
    "AUROC": {
        "rows": ["NF (MIMIC-IV)"],
        "mean": np.array([[np.nan, np.nan, 0.764, 0.743, 0.783]]),
        "std": np.array([[np.nan, np.nan, 0.038, 0.037, np.nan]]),
    },
    "AUPRC": {
        "rows": ["NF (MIMIC-IV)"],
        "mean": np.array([[np.nan, np.nan, 0.166, 0.131, 0.199]]),
        "std": np.array([[np.nan, np.nan, 0.059, 0.035, np.nan]]),
    },
    "F1 score": {
        "rows": ["NF (MIMIC-IV)"],
        "mean": np.array([[np.nan, np.nan, 0.183, 0.168, 0.137]]),
        "std": np.array([[np.nan, np.nan, 0.020, 0.017, np.nan]]),
    },
}

METRICS = ["AUROC", "AUPRC", "F1 score"]
LETTERS = ["a", "b", "c"]
NORM = TwoSlopeNorm(vmin=-0.06, vcenter=0.0, vmax=0.06)
CMAP = plt.cm.RdBu


def _cell_text(m: float, s: float) -> str:
    if np.isnan(m):
        return "—"
    if np.isnan(s):
        return f"{m:.3f}"
    return f"{m:.3f}\n±{s:.3f}"


def _add_bottom_bands(ax, n_rows: int, n_cols: int = 5) -> None:
    """Paper-style Pretraining / Finetuning strips *under* the heatmap."""
    # imshow y increases downward; bottom edge of grid is at n_rows - 0.5
    y0 = n_rows - 0.5
    y_pt = y0 + 0.18
    y_ft = y0 + 0.68
    h = 0.42

    for (lab, c0, c1), color in zip(COL_LABELS_PT, PT_COLORS):
        ax.add_patch(
            Rectangle(
                (c0 - 0.5, y_pt),
                c1 - c0,
                h,
                facecolor=color,
                edgecolor="0.45",
                linewidth=0.6,
                clip_on=False,
                zorder=5,
            )
        )
        ax.text(
            (c0 + c1) / 2 - 0.5,
            y_pt + h / 2,
            lab,
            ha="center",
            va="center",
            fontsize=7.5,
            clip_on=False,
            zorder=6,
        )

    for j, (lab, color) in enumerate(zip(COL_LABELS_FT, FT_COLORS)):
        ax.add_patch(
            Rectangle(
                (j - 0.5, y_ft),
                1,
                h,
                facecolor=color,
                edgecolor="0.45",
                linewidth=0.6,
                clip_on=False,
                zorder=5,
            )
        )
        ax.text(
            j,
            y_ft + h / 2,
            lab,
            ha="center",
            va="center",
            fontsize=7.5,
            clip_on=False,
            zorder=6,
        )


def _draw_panel(ax, metric: str, letter: str, block: dict, show_ylabels: bool) -> None:
    means = block["mean"]
    stds = block["std"]
    rows = block["rows"]
    n_rows, n_cols = means.shape

    baselines = np.array([BASE[metric][r] for r in rows], dtype=float)[:, None]
    diff = means - baselines
    # baseline column: no color (neutral)
    diff[:, -1] = 0.0
    # missing cells: mask
    mask = np.isnan(means)
    diff_m = np.ma.array(diff, mask=mask)

    ax.imshow(diff_m, cmap=CMAP, norm=NORM, aspect="equal", interpolation="nearest")

    # grid
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="0.55", lw=0.7, zorder=4)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="0.55", lw=0.7, zorder=4)

    # gray overlay on none/none column
    for i in range(n_rows):
        ax.add_patch(
            Rectangle(
                (n_cols - 1.5, i - 0.5),
                1,
                1,
                facecolor="#c8c8c8",
                edgecolor="0.55",
                lw=0.7,
                zorder=3,
            )
        )

    # annotations
    for i in range(n_rows):
        for j in range(n_cols):
            txt = _cell_text(means[i, j], stds[i, j])
            fs = 6.0 if "\n" in txt else 7.5
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=fs,
                color="black",
                zorder=7,
                linespacing=0.95,
            )

    ax.set_xticks([])
    ax.set_yticks(range(n_rows))
    if show_ylabels:
        ax.set_yticklabels(rows, fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=2)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # title like paper: bold letter + metric name above panel
    ax.set_title(f"{letter}    {metric}", loc="left", fontsize=11, fontweight="bold", pad=8)

    _add_bottom_bands(ax, n_rows=n_rows, n_cols=n_cols)


def make_figure(data: dict, out_stem: Path, figsize_h: float) -> None:
    n_rows = len(next(iter(data.values()))["rows"])
    fig_w = 11.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_w, figsize_h),
        gridspec_kw={"wspace": 0.22},
    )

    for ax, metric, letter in zip(axes, METRICS, LETTERS):
        _draw_panel(ax, metric, letter, data[metric], show_ylabels=(ax is axes[0]))

    # left labels for category bands (first panel only)
    n0 = len(next(iter(data.values()))["rows"])
    y0 = n0 - 0.5
    axes[0].text(
        -0.85,
        y0 + 0.18 + 0.21,
        "Pretraining",
        ha="right",
        va="center",
        fontsize=7.5,
        clip_on=False,
        transform=axes[0].transData,
    )
    axes[0].text(
        -0.85,
        y0 + 0.68 + 0.21,
        "Finetuning",
        ha="right",
        va="center",
        fontsize=7.5,
        clip_on=False,
        transform=axes[0].transData,
    )

    # shared colorbar
    cax = fig.add_axes([0.915, 0.34, 0.015, 0.42])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
        cax=cax,
    )
    cb.set_ticks([-0.06, -0.03, 0.0, 0.03, 0.06])
    cb.set_label("difference", fontsize=9)
    cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.18, right=0.90, top=0.88, bottom=0.28)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: script dir / figures)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "figures")
    out_dir = out_dir.resolve()

    make_figure(PRIMENET, out_dir / "fig_primenet_fig5", figsize_h=3.85)
    make_figure(STRATS, out_dir / "fig_strats_fig5", figsize_h=2.85)
    print(f"Done. Upload PDF/PNG from: {out_dir}")


if __name__ == "__main__":
    main()
