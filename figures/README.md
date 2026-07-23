# Figures

Paper-style **STraTS Fig. 5** heatmaps for the project report.

| File | Content |
|------|---------|
| `fig_primenet_fig5.pdf` / `.png` | PrimeNet on MIMIC-IV NF + aplasia (5-fold mean ± std) |
| `fig_strats_fig5.pdf` / `.png` | STraTS on MIMIC-IV NF in-domain |

## Regenerate

From the repo root (with `matplotlib` and `numpy` installed):

```bash
export PYTHONPATH="$(pwd)"
python scripts/plot_fig5_heatmaps.py --out-dir figures
```

Numbers are hardcoded from verified `results_final.csv` means/stds in the script.
Update the matrices in `scripts/plot_fig5_heatmaps.py` after new runs.

## Coloring

Cell color = **mean − published ChemoTree STraTS `none/none` baseline** (same row), diverging RdBu, clipped to ±0.06. Gray column = baseline mean only (PrimeNet `none/none` not run in the reported matrix).
