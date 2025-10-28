#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
new_grid_search.py

Runs the ablation grid described in train.py (LN, Bias, freeze_wv, freeze_wo
at d_model=64, n_heads=1, n_layers=2) plus the extra fixed configs over
d_model in [128, 32, 8] with LN=False, Bias=False, freeze_wv=True, freeze_wo=True.

Training/eval pipeline follows listlen_grid_search.py (synthetic batches), and
results are appended to a CSV (default: ablation_res.csv) with POSIX file locks
and skip-existing behavior.
"""

import os
import time
import csv
import argparse
import hashlib
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import fcntl  # POSIX file locking

from transformer_lens import HookedTransformer
from model_utils import configure_runtime, make_model


# -----------------------------
# Device, seeds, utils
# -----------------------------


def pick_device(explicit: str = "auto") -> torch.device:
    if explicit != "auto":
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_batch(
    batch_size: int,
    n_digits: int,
    list_len: int,
    device: torch.device,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      inputs  [B, T]
      targets [B, T]
    where T = 2L + 1 and evaluation uses targets[:, L+1:].
    """
    pad = n_digits
    sep = n_digits + 1
    T = 2 * list_len + 1

    digits = torch.randint(
        0, n_digits, (batch_size, list_len), generator=rng, device=device
    )

    inputs = torch.full((batch_size, T), pad, dtype=torch.long, device=device)
    targets = torch.full((batch_size, T), sep, dtype=torch.long, device=device)

    inputs[:, :list_len] = digits
    inputs[:, list_len] = sep

    targets[:, :list_len] = digits
    targets[:, list_len] = sep
    targets[:, list_len + 1 :] = digits
    return inputs, targets


def make_validation_set(
    n_examples: int,
    n_digits: int,
    list_len: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    X, Y = make_batch(n_examples, n_digits, list_len, device, g)
    return X, Y


@dataclass(frozen=True)
class Config:
    LIST_LEN: int
    N_DIGITS: int
    D_MODEL: int
    N_HEAD: int
    N_LAYERS: int
    USE_LN: bool
    USE_BIAS: bool
    FREEZE_WV: bool
    FREEZE_WO: bool
    WEIGHT_DECAY: float
    run_idx: int


def eval_accuracy(
    model: nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    list_len: int,
    batch_size: int = 1024,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    hits = 0
    tots = 0
    with torch.no_grad():
        for i in range(0, val_inputs.size(0), batch_size):
            xb = val_inputs[i : i + batch_size].to(device)
            yb = val_targets[i : i + batch_size].to(device)
            logits = model(xb)[:, list_len + 1 :, :]  # [B, L, V]
            preds = logits.argmax(dim=-1)
            gold = yb[:, list_len + 1 :]
            hits += (preds == gold).sum().item()
            tots += preds.numel()
    return hits / max(1, tots)


def build_model_from_config(cfg: Config, device: torch.device) -> nn.Module:
    assert cfg.D_MODEL % cfg.N_HEAD == 0, "d_model must be divisible by n_heads"
    vocab = cfg.N_DIGITS + 2
    seq_len = 2 * cfg.LIST_LEN + 1
    # Configure shared runtime and build via model_utils
    configure_runtime(list_len=cfg.LIST_LEN, seq_len=seq_len, vocab=vocab, device=device)
    model = make_model(
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEAD,
        d_model=cfg.D_MODEL,
        ln=cfg.USE_LN,
        use_bias=cfg.USE_BIAS,
        freeze_wv=cfg.FREEZE_WV,
        freeze_wo=cfg.FREEZE_WO,
        device=device,
    )
    return model


def train_and_return_model(
    cfg: Config,
    device: torch.device,
    train_steps: int,
    batch_size: int,
    eval_every: int,
    early_stop_acc: float,
    val_size: int,
    base_seed: int,
    lr: float,
    eval_batch_size: int,
    grad_clip: float,
) -> Tuple[nn.Module, Dict]:
    seed_material = (str(asdict(cfg)) + str(base_seed)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    set_global_seed(seed + cfg.run_idx)

    model = build_model_from_config(cfg, device)
    total_params, trainable_params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    val_inputs, val_targets = make_validation_set(
        n_examples=val_size,
        n_digits=cfg.N_DIGITS,
        list_len=cfg.LIST_LEN,
        device=device,
        seed=seed + 123456,
    )

    model.train()
    g = torch.Generator(device=device).manual_seed(seed + 999)

    last_eval = 0.0
    steps_run = 0
    start = time.time()

    pbar = tqdm(total=train_steps, desc="train", leave=False, ncols=120)
    for step in range(1, train_steps + 1):
        xb, yb = make_batch(batch_size, cfg.N_DIGITS, cfg.LIST_LEN, device, g)
        logits = model(xb)[:, cfg.LIST_LEN + 1 :]
        gold = yb[:, cfg.LIST_LEN + 1 :]

        loss = ce(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad()

        if step % eval_every == 0 or step == train_steps:
            last_eval = eval_accuracy(
                model, val_inputs, val_targets, cfg.LIST_LEN, batch_size=eval_batch_size
            )
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                train_acc = (preds == gold).float().mean().item()

            pbar.set_postfix(
                {"val_acc": f"{last_eval:.4f}", "train_acc": f"{train_acc:.4f}"}
            )
            if last_eval >= early_stop_acc:
                steps_run = step
                pbar.update(train_steps - step + 1)
                break

        pbar.update(1)
        steps_run = step
    pbar.close()

    train_time_sec = time.time() - start
    if last_eval == 0.0:
        last_eval = eval_accuracy(
            model, val_inputs, val_targets, cfg.LIST_LEN, batch_size=eval_batch_size
        )

    metrics = {
        "val_acc": last_eval,
        "steps_trained": steps_run,
        "train_time_sec": train_time_sec,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "val_examples": val_inputs.size(0),
        "batch_size": batch_size,
    }
    return model, metrics


def train_one(
    cfg: Config,
    device: torch.device,
    train_steps: int,
    batch_size: int,
    eval_every: int,
    early_stop_acc: float,
    val_size: int,
    base_seed: int,
    lr: float,
    eval_batch_size: int,
    grad_clip: float,
) -> Dict:
    _model, metrics = train_and_return_model(
        cfg=cfg,
        device=device,
        train_steps=train_steps,
        batch_size=batch_size,
        eval_every=eval_every,
        early_stop_acc=early_stop_acc,
        val_size=val_size,
        base_seed=base_seed,
        lr=lr,
        eval_batch_size=eval_batch_size,
        grad_clip=grad_clip,
    )
    return metrics


# -----------------------------
# Ablation grids
# -----------------------------


def build_ablation_grid(n_runs: int) -> List[Config]:
    """Grid over LN, Bias, freeze_wv, freeze_wo at fixed d/h/L.

    Mirrors the section in train.py with defaults:
      d_model=64, n_heads=1, n_layers=2, LIST_LEN=2, N_DIGITS=100.
    """
    GRID_D_MODEL = 64
    GRID_N_HEADS = 1
    GRID_N_LAYERS = 2
    LIST_LEN = 2
    N_DIGITS = 100
    WEIGHT_DECAY = 0.01

    grid_lns = [False, True]
    grid_biases = [False, True]
    grid_freeze_wv = [False, True]
    grid_freeze_wo = [False, True]

    cfgs: List[Config] = []
    for ln in grid_lns:
        for bias in grid_biases:
            for fwv in grid_freeze_wv:
                for fwo in grid_freeze_wo:
                    for run_idx in range(1, n_runs + 1):
                        cfgs.append(
                            Config(
                                LIST_LEN=LIST_LEN,
                                N_DIGITS=N_DIGITS,
                                D_MODEL=GRID_D_MODEL,
                                N_HEAD=GRID_N_HEADS,
                                N_LAYERS=GRID_N_LAYERS,
                                USE_LN=ln,
                                USE_BIAS=bias,
                                FREEZE_WV=fwv,
                                FREEZE_WO=fwo,
                                WEIGHT_DECAY=0.01,
                                run_idx=run_idx,
                            )
                        )
    return cfgs


def build_extra_grid(n_runs: int, d_models: List[int]) -> List[Config]:
    """Fixed configs over d_model with LN/Bias=False and fWV/fWO=True."""
    LIST_LEN = 2
    N_DIGITS = 100
    GRID_N_HEADS = 1
    GRID_N_LAYERS = 2
    WEIGHT_DECAY = 0.01

    cfgs: List[Config] = []
    for d in d_models:
        for run_idx in range(1, n_runs + 1):
            cfgs.append(
                Config(
                    LIST_LEN=LIST_LEN,
                    N_DIGITS=N_DIGITS,
                    D_MODEL=d,
                    N_HEAD=GRID_N_HEADS,
                    N_LAYERS=GRID_N_LAYERS,
                    USE_LN=False,
                    USE_BIAS=False,
                    FREEZE_WV=True,
                    FREEZE_WO=True,
                    WEIGHT_DECAY=WEIGHT_DECAY,
                    run_idx=run_idx,
                )
            )
    return cfgs


# -----------------------------
# CSV helpers
# -----------------------------


def ensure_csv_header(path: str, fields: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        empty = f.tell() == 0
        if empty:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def append_row_locked(path: str, fields: List[str], row: Dict) -> None:
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


# -----------------------------
# CLI and main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation Grid Search for Copy Task")
    p.add_argument("--output", type=str, default="ablation_res.csv", help="CSV file path")
    p.add_argument("--device", type=str, default="auto", help="cuda|mps|cpu|auto")
    p.add_argument("--seed", type=int, default=42, help="base seed")
    p.add_argument("--train-steps", type=int, default=50000, help="max steps per run")
    p.add_argument("--eval-every", type=int, default=2500, help="validation frequency")
    p.add_argument("--early-stop-acc", type=float, default=0.999, help="early stop threshold")
    p.add_argument("--val-size", type=int, default=8192, help="validation set size")
    p.add_argument("--batch-size", type=int, default=1024, help="training batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--eval-batch-size", type=int, default=1024, help="evaluation batch size")
    p.add_argument("--grad-clip", type=float, default=0.0, help="max grad-norm clipping (0 disables)")
    p.add_argument("--skip-existing", action="store_true", default=True, help="skip configs already present in output CSV")
    p.add_argument("--runs", type=int, default=30, help="number of runs per config")
    p.add_argument(
        "--extra-d-models",
        type=str,
        default="128,32,8",
        help="comma-separated list of extra d_model values (with LN/Bias False and fWV/fWO True)",
    )
    p.add_argument("--no-extra", action="store_true", help="disable running the extra d_model configs")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    set_global_seed(args.seed)

    # Build config list
    cfgs = build_ablation_grid(n_runs=args.runs)
    if not args.no_extra:
        extra_dims = [int(x) for x in args.extra_d_models.split(",") if x.strip()]
        cfgs += build_extra_grid(n_runs=args.runs, d_models=extra_dims)

    total = len(cfgs)

    fields = [
        "LIST_LEN",
        "N_DIGITS",
        "D_MODEL",
        "N_HEAD",
        "N_LAYERS",
        "USE_LN",
        "USE_BIAS",
        "FREEZE_WV",
        "FREEZE_WO",
        "WEIGHT_DECAY",
        "run_idx",
        "val_acc",
        "steps_trained",
        "train_time_sec",
        "params_total",
        "params_trainable",
        "val_examples",
        "batch_size",
        "device",
    ]
    ensure_csv_header(args.output, fields)

    # Optionally load existing rows to skip duplicates
    existing_keys = set()
    if args.skip_existing and os.path.exists(args.output):
        try:
            with open(args.output, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (
                        int(row["LIST_LEN"]),
                        int(row["N_DIGITS"]),
                        int(row["D_MODEL"]),
                        int(row["N_HEAD"]),
                        int(row["N_LAYERS"]),
                        row["USE_LN"] in ("True", "true", "1"),
                        row["USE_BIAS"] in ("True", "true", "1"),
                        row["FREEZE_WV"] in ("True", "true", "1"),
                        row["FREEZE_WO"] in ("True", "true", "1"),
                        float(row["WEIGHT_DECAY"]),
                        int(row["run_idx"]),
                    )
                    existing_keys.add(key)
        except Exception:
            pass

    outer = tqdm(total=total, desc="ablation", ncols=100)
    for cfg in cfgs:
        if args.skip_existing:
            key = (
                cfg.LIST_LEN,
                cfg.N_DIGITS,
                cfg.D_MODEL,
                cfg.N_HEAD,
                cfg.N_LAYERS,
                cfg.USE_LN,
                cfg.USE_BIAS,
                cfg.FREEZE_WV,
                cfg.FREEZE_WO,
                cfg.WEIGHT_DECAY,
                cfg.run_idx,
            )
            if key in existing_keys:
                outer.update(1)
                continue
        try:
            metrics = train_one(
                cfg=cfg,
                device=device,
                train_steps=args.train_steps,
                batch_size=args.batch_size,
                eval_every=args.eval_every,
                early_stop_acc=args.early_stop_acc,
                val_size=args.val_size,
                base_seed=args.seed,
                lr=args.lr,
                eval_batch_size=args.eval_batch_size,
                grad_clip=args.grad_clip,
            )
        except RuntimeError:
            metrics = {
                "val_acc": float("nan"),
                "steps_trained": 0,
                "train_time_sec": 0.0,
                "params_total": 0,
                "params_trainable": 0,
                "val_examples": args.val_size,
                "batch_size": args.batch_size,
            }

        row = {**asdict(cfg), **metrics, "device": str(device)}
        append_row_locked(args.output, fields, row)
        outer.update(1)
    outer.close()

    print(f"Wrote {total} rows to {args.output}")


if __name__ == "__main__":
    main()
