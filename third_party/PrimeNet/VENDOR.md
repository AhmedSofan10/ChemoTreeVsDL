# Vendored PrimeNet (TimeBERT)

Upstream: https://github.com/ranakroychowdhury/PrimeNet  
Pinned commit: `454386b` (local vendor snapshot, June 2026)

Included files (training runtime only):

- `timebert/` — encoder and pretrain/finetune heads
- `collator.py` — TimeCL / TimeReco data augmentation
- `pretrain.py` — pretrain CLI defaults and train/eval loops
- `utils.py` — dataloaders and irregular sample helpers

Not vendored: datasets, preprocess scripts, `finetune.py` (ChemoTree uses `ts_model_training/primenet/train_loop.py`).

License: see upstream repository (AAAI 2023 publication).
