"""PrimeNet pretrain + finetune using the official TimeBERT code."""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ts_model_training.primenet.paths import get_primenet_root, primenet_tensor_dir


def _ensure_primenet_imports():
    root = get_primenet_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _import_pretrain_module():
    """PrimeNet pretrain.py parses sys.argv at import time; isolate it from our CLI."""
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]]
    try:
        import pretrain as pn_pretrain
        return pn_pretrain
    finally:
        sys.argv = saved_argv


def _configure_pretrain_args(pn_pretrain, tensor_dir: Path, params: dict):
    """Reuse upstream defaults (segment_num, mask_ratio_per_seg, …) and override paths."""
    args = pn_pretrain.args
    args.path = str(tensor_dir / "pretrain") + "/"
    args.pretrain_tasks = params.get("pretrain_tasks", args.pretrain_tasks)
    args.niters = int(params.get("pretrain_niters", args.niters))
    args.batch_size = int(params.get("batch_size", args.batch_size))
    args.lr = float(params.get("lr", args.lr))
    args.rec_hidden = int(params.get("rec_hidden", args.rec_hidden))
    args.embed_time = int(params.get("embed_time", args.embed_time))
    args.num_heads = int(params.get("num_heads", args.num_heads))
    args.learn_emb = True
    args.pooling = params.get("pretrain_pooling", params.get("pooling", args.pooling))
    args.patience = int(params.get("patience", args.patience))
    args.seed = int(params.get("seed", args.seed))
    args.dev = str(params.get("dev", args.dev))
    if "segment_num" in params:
        args.segment_num = int(params["segment_num"])
    if "mask_ratio_per_seg" in params:
        args.mask_ratio_per_seg = float(params["mask_ratio_per_seg"])
    args.device = _device(args.dev)
    return args


def _torch_load_pt(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _install_colab_primenet_utils_patches(params: dict) -> None:
    """Avoid upstream num_workers=8 (Colab freeze) and optional subsample for --fast."""
    import utils as pn_utils
    from collator import CLDataCollator

    if getattr(pn_utils, "_bionets_colab_patch", False):
        return

    max_pre = params.get("max_pretrain_samples")
    max_ft = params.get("max_finetune_samples")

    def generate_batches(X_train, X_val, args):
        input_dim = (X_train.shape[2] - 1) // 2
        X_train, train_max_len = pn_utils.generate_irregular_samples(X_train, input_dim)
        X_val, val_max_len = pn_utils.generate_irregular_samples(X_val, input_dim)
        max_len = max(train_max_len, val_max_len)
        pretrain_data = pn_utils.TimeDataset(X_train)
        val_data = pn_utils.TimeDataset(X_val)
        collator = CLDataCollator(max_len=max_len, args=args)
        batch_size = min(min(len(val_data), args.batch_size), args.n)
        train_dataloader = DataLoader(
            pretrain_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        return {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "input_dim": input_dim,
            "max_len": max_len,
            "n_train_batches": len(train_dataloader),
            "n_test_batches": len(val_dataloader),
        }

    def get_unlabeled_pretrain_data(args):
        X_train = _torch_load_pt(args.path + "X_train.pt")
        X_val = _torch_load_pt(args.path + "X_val.pt")
        if max_pre is not None:
            X_train = X_train[: int(max_pre)]
            X_val = X_val[: max(64, int(max_pre) // 5)]
        print(f"X_train: {X_train.shape}", flush=True)
        print(f"X_val: {X_val.shape}", flush=True)
        return generate_batches(X_train, X_val, args)

    def get_finetune_data(args):
        X_train = _torch_load_pt(args.path + "X_train.pt")
        y_train = _torch_load_pt(args.path + "y_train.pt")
        X_val = _torch_load_pt(args.path + "X_val.pt")
        y_val = _torch_load_pt(args.path + "y_val.pt")
        X_test = _torch_load_pt(args.path + "X_test.pt")
        y_test = _torch_load_pt(args.path + "y_test.pt")
        if max_ft is not None:
            n = int(max_ft)
            X_train, y_train = X_train[:n], y_train[:n]
            X_val, y_val = X_val[: n // 4], y_val[: n // 4]
            X_test, y_test = X_test[: n // 4], y_test[: n // 4]
        input_dim = (X_train.shape[2] - 1) // 2
        print(
            f"finetune X_train {X_train.shape} y_train {y_train.shape}",
            flush=True,
        )
        train_ds = TensorDataset(X_train, y_train.long().squeeze())
        val_ds = TensorDataset(X_val, y_val.long().squeeze())
        test_ds = TensorDataset(X_test, y_test.long().squeeze())
        kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
        return {
            "train_dataloader": DataLoader(train_ds, shuffle=True, **kw),
            "val_dataloader": DataLoader(val_ds, shuffle=False, **kw),
            "test_dataloader": DataLoader(test_ds, shuffle=False, **kw),
            "input_dim": input_dim,
        }

    pn_utils.generate_batches = generate_batches
    pn_utils.get_unlabeled_pretrain_data = get_unlabeled_pretrain_data
    pn_utils.get_finetune_data = get_finetune_data
    pn_utils._bionets_colab_patch = True


def _device(dev: str = "0") -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dev}")
    return torch.device("cpu")


def run_pretrain(
    tensor_dir: Path,
    ckpt_path: Path,
    params: dict,
    logger=None,
) -> None:
    _ensure_primenet_imports()
    import utils as pn_utils

    pn_pretrain = _import_pretrain_module()
    _install_colab_primenet_utils_patches(params)
    args = _configure_pretrain_args(pn_pretrain, tensor_dir, params)
    from timebert import TimeBERTConfig, TimeBERTForPretrainingV2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_obj = pn_utils.get_unlabeled_pretrain_data(args)
    dim = data_obj["input_dim"]
    max_len = max(data_obj["max_len"], 512)
    print(
        f"Pretrain: {args.niters} epochs, batch_size={args.batch_size}, "
        f"batches/epoch={data_obj['n_train_batches']}",
        flush=True,
    )

    config = TimeBERTConfig(
        input_dim=dim,
        pretrain_tasks=args.pretrain_tasks,
        cls_query=torch.linspace(0, 1.0, 128),
        hidden_size=args.rec_hidden,
        embed_time=args.embed_time,
        num_heads=args.num_heads,
        learn_emb=args.learn_emb,
        freq=args.freq,
        pooling=args.pooling,
        max_length=max_len,
        dropout=0.3,
        temp=0.05,
    )
    model = TimeBERTForPretrainingV2(config).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    patience = args.patience
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for itr in range(1, args.niters + 1):
        t0 = time.time()
        print(f"pretrain epoch {itr}/{args.niters} ...", flush=True)
        pn_pretrain.train(args, model, data_obj["train_dataloader"], optimizer)
        _, _, val_acc = pn_pretrain.eval(args, model, data_obj["val_dataloader"])
        print(
            f"  epoch {itr} done in {time.time() - t0:.0f}s  val_acc={val_acc:.4f}  best={max(best_acc, val_acc):.4f}",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "args": args,
                    "epoch": itr,
                    "model_state_dict": model.bert.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            patience = args.patience
        else:
            patience -= 1
            if patience < 0:
                break
    if logger:
        logger.write(f"Pretrain checkpoint -> {ckpt_path} (best val_acc={best_acc:.4f})")


def run_finetune(
    tensor_dir: Path,
    ckpt_path: Path,
    params: dict,
    logger=None,
) -> dict:
    from ts_model_training.primenet.metrics import evaluate_primenet_classifier

    _ensure_primenet_imports()
    import utils as pn_utils

    _install_colab_primenet_utils_patches(params)
    from timebert import TimeBERTConfig, TimeBERTForClassification

    args = Namespace(
        path=str(tensor_dir / "finetune") + "/",
        dataset="MIMIC-III",
        task="classification",
        classif=True,
        classify_pertp=False,
        niters=int(params.get("finetune_niters", 2000)),
        batch_size=int(params.get("batch_size", 64)),
        lr=float(params.get("lr", 0.0001)),
        rec_hidden=int(params.get("rec_hidden", 128)),
        embed_time=int(params.get("embed_time", 128)),
        num_heads=int(params.get("num_heads", 1)),
        learn_emb=True,
        freq=10.0,
        pooling=params.get("finetune_pooling", params.get("pooling", "ave")),
        seed=int(params.get("seed", 0)),
        dev=str(params.get("dev", "0")),
        save=0,
        n=8000,
        patience=int(params.get("finetune_patience", params.get("patience", 20))),
    )
    args.device = _device(args.dev)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_obj = pn_utils.get_finetune_data(args)
    dim = data_obj["input_dim"]

    config = TimeBERTConfig(
        dataset=args.dataset,
        input_dim=dim,
        cls_query=torch.linspace(0, 1.0, 128),
        hidden_size=args.rec_hidden,
        embed_time=args.embed_time,
        num_heads=args.num_heads,
        learn_emb=args.learn_emb,
        freq=args.freq,
        pooling=args.pooling,
        classify_pertp=args.classify_pertp,
        max_length=512,
        dropout=0.3,
        temp=0.05,
    )
    model = TimeBERTForClassification(config).to(args.device)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing pretrain checkpoint: {ckpt_path}")
    model.bert.load_state_dict(_torch_load_pt(str(ckpt_path))["model_state_dict"])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_state = None
    patience = args.patience
    finetune_ckpt = ckpt_path.parent / "primenet_finetune_best.pt"

    print(
        f"Finetune: max {args.niters} epochs, batch_size={args.batch_size}, "
        f"pooling={args.pooling}, train_batches={len(data_obj['train_dataloader'])}",
        flush=True,
    )
    for itr in range(1, args.niters + 1):
        t0 = time.time()
        print(f"finetune epoch {itr}/{args.niters} ...", flush=True)
        model.train()
        for train_batch, label in data_obj["train_dataloader"]:
            train_batch = train_batch.to(args.device)
            label = label.to(args.device)
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim : 2 * dim]
            observed_tp = train_batch[:, :, -1]
            out = model(torch.cat((observed_data, observed_mask), 2), observed_tp)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_primenet_classifier(
            model, data_obj["val_dataloader"], args, dim
        )
        print(
            f"  epoch {itr} done in {time.time() - t0:.0f}s  "
            f"val_auroc={val_metrics['auroc']:.4f}",
            flush=True,
        )
        if val_metrics["auroc"] > best_val_auc:
            best_val_auc = val_metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, finetune_ckpt)
            patience = args.patience
        else:
            patience -= 1
            if patience < 0:
                print(f"Finetune early stop at epoch {itr}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_primenet_classifier(
        model, data_obj["test_dataloader"], args, dim
    )
    print(
        f"Test (best val checkpoint): auroc={test_metrics['auroc']:.4f} "
        f"auprc={test_metrics['auprc']:.4f} f1={test_metrics['f1']:.4f}",
        flush=True,
    )
    if logger:
        logger.write(f"Best val auroc: {best_val_auc:.4f}")
        logger.write(f"Test metrics: {test_metrics}")
    return test_metrics


def train_fold(
    cohort: str,
    fold: int,
    params: dict,
    output_path: Path,
    skip_export: bool = False,
    skip_pretrain: bool = False,
) -> dict:
    from ts_model_training.logger import Logger
    from ts_model_training.primenet.export import export_fold

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_path, "log.txt")
    logger.write("PrimeNet fold training (MIMIC-IV export)")

    tensor_dir = primenet_tensor_dir(cohort, fold)
    if not skip_export and not (tensor_dir / "finetune" / "X_train.pt").is_file():
        export_fold(
            cohort,
            fold,
            max_obs=int(params.get("max_obs", 512)),
            days_before_discharge=int(params.get("days_before_discharge", 14)),
        )

    ckpt_path = output_path / "primenet_pretrain.h5"
    if skip_pretrain and ckpt_path.is_file():
        logger.write(f"Skipping pretrain; using {ckpt_path}")
    else:
        run_pretrain(tensor_dir, ckpt_path, params, logger=logger)
    metrics = run_finetune(tensor_dir, ckpt_path, params, logger=logger)

    final = {
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "loss": metrics["loss"],
    }
    logger.write(f"Final test res: {final}")
    return final
