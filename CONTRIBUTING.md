# Contributing

Thanks for interest in the `primenet` branch.

## Development

1. Branch from `primenet`.
2. Keep changes small and focused (easier for supervisors to review).
3. Do not commit MIMIC raw data, `saved_data/`, or checkpoints.
4. Run at least:

```bash
export PYTHONPATH="$(pwd)"
python scripts/run_primenet_fig5.py --help
python scripts/plot_fig5_heatmaps.py --help
python scripts/collect_primenet_results.py --help
```

## Code style

- Match existing ChemoTree module layout (`ts_model_training/`, `scripts/`).
- Prefer clear CLI `--help` and docstring examples on new scripts.
- Update `docs/REPRODUCE.md` when you change the training workflow.

## Issues

Use GitHub Issues for bugs and reproduction failures. Include OS, GPU, and the exact command.
