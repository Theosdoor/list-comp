#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter sweep to achieve 100% accuracy with n_layers <= list_len
Uses Bayesian-inspired sampling for efficient exploration

What it does:

1. For each list_len, creates models with n_layers = list_len
2. Samples hyperparameters (lr, weight_decay, scheduler, grad_clip) using chosen strategy
3. Trains each config and saves results to CSV
4. Early stops at 100% accuracy or after patience
5. CSV is append-safe with file locking (can run multiple in parallel)
"""
import os
import time
import csv
import argparse
import hashlib
import random
import math
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

import fcntl  # POSIX file locking

import sys
sys.path.insert(0, '../..')
from transformer_lens import HookedTransformer
from gridsearch_utils import (
    pick_device,
    set_global_seed,
    count_params,
    make_batch,
    make_validation_set,
    eval_accuracy,
    build_model,
)


@dataclass(frozen=True)
class Config:
    # Architecture
    list_len: int
    n_digits: int = 100  # Fixed default
    d_model: int = 64    # Fixed default
    n_heads: int = 1     # Fixed default
    n_layers: int = 2
    
    # Training hyperparameters (to sweep)
    lr: float = 1e-3
    weight_decay: float = 0.01
    use_lr_scheduler: bool = False
    warmup_steps: int = 1000
    max_grad_norm: Optional[float] = None
    
    # Fixed params
    max_steps: int = 100000
    batch_size: int = 2048
    seed: int = 0


def build_model_from_config(cfg: Config, device: torch.device) -> HookedTransformer:
    """Build model from config dataclass"""
    return build_model(
        list_len=cfg.list_len,
        n_digits=cfg.n_digits,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        use_ln=False,  # Fixed per requirements
        use_bias=False,  # Fixed
        use_wv=False,  # Fixed
        use_wo=False,  # Fixed
        attn_only=True,  # Fixed (no MLP)
        device=device,
        seed=cfg.seed,
    )


def train_and_evaluate(
    cfg: Config,
    device: torch.device,
    val_size: int = 8192,
    eval_every: int = 100,
    early_stop_acc: float = 1.0,
    early_stop_threshold: float = 0.9,
    patience: int = 50,
) -> Dict:
    """Train model with LR scheduler and early stopping"""
    set_global_seed(cfg.seed)
    
    model = build_model_from_config(cfg, device)
    total_params, trainable_params = count_params(model)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Setup LR scheduler (warmup + cosine decay)
    scheduler = None
    if cfg.use_lr_scheduler:
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < cfg.warmup_steps:
                return float(current_step) / float(max(1, cfg.warmup_steps))
            # Cosine decay phase
            progress = float(current_step - cfg.warmup_steps) / float(max(1, cfg.max_steps - cfg.warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(opt, lr_lambda)
    
    ce = nn.CrossEntropyLoss()
    
    # Validation set
    val_inputs, val_targets = make_validation_set(
        n_examples=val_size,
        n_digits=cfg.n_digits,
        list_len=cfg.list_len,
        device=device,
        seed=cfg.seed + 123456,
    )
    
    model.train()
    g = torch.Generator(device=device).manual_seed(cfg.seed + 999)
    
    best_acc = 0.0
    steps_without_improvement = 0
    steps_run = 0
    start = time.time()
    
    pbar = tqdm(total=cfg.max_steps, desc=f"L{cfg.list_len}N{cfg.n_layers}", leave=False, ncols=140)
    
    for step in range(1, cfg.max_steps + 1):
        xb, yb = make_batch(cfg.batch_size, cfg.n_digits, cfg.list_len, device, g)
        logits = model(xb)[:, cfg.list_len + 1:]
        gold = yb[:, cfg.list_len + 1:]
        
        loss = ce(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        loss.backward()
        
        # Gradient clipping
        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        
        opt.step()
        opt.zero_grad()
        
        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()
        
        if step % eval_every == 0 or step == cfg.max_steps:
            acc = eval_accuracy(model, val_inputs, val_targets, cfg.list_len, batch_size=1024)
            current_lr = opt.param_groups[0]['lr']
            
            # Update progress bar
            postfix = {
                "acc": f"{acc:.4f}",
                "best": f"{best_acc:.4f}",
                "loss": f"{loss.item():.4f}",
            }
            if scheduler is not None:
                postfix["lr"] = f"{current_lr:.2e}"
            pbar.set_postfix(postfix)
            
            # Early stopping at 100%
            if acc >= early_stop_acc:
                steps_run = step
                pbar.update(cfg.max_steps - step + 1)
                break
            
            # Patience-based early stopping (only if acc > threshold)
            if acc > best_acc:
                best_acc = acc
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if acc >= early_stop_threshold and steps_without_improvement >= patience:
                    steps_run = step
                    pbar.update(cfg.max_steps - step + 1)
                    break
        
        pbar.update(1)
        steps_run = step
    
    pbar.close()
    
    train_time_sec = time.time() - start
    final_acc = eval_accuracy(model, val_inputs, val_targets, cfg.list_len, batch_size=1024)
    
    metrics = {
        "val_acc": final_acc,
        "steps_trained": steps_run,
        "train_time_sec": train_time_sec,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "val_examples": val_size,
    }
    return metrics


def sample_hyperparams(run_idx: int, n_samples: int, strategy: str = "random") -> Dict[str, float]:
    """Sample hyperparameters using different strategies"""
    rng = np.random.RandomState(run_idx)
    
    if strategy == "random":
        # Random sampling
        lr = 10 ** rng.uniform(-4, np.log10(5e-3))
        weight_decay = 10 ** rng.uniform(-5, -1)
        max_grad_norm = rng.uniform(0.5, 2.0) if rng.rand() > 0.2 else None
        use_lr_scheduler = rng.rand() > 0.5
    
    elif strategy == "grid":
        # Coarse grid
        lrs = [5e-4, 1e-3, 2e-3]
        wds = [0.0, 0.001, 0.01]
        schedulers = [True, False]
        grad_norms = [None, 1.0]
        
        # Enumerate all combinations
        idx = run_idx % (len(lrs) * len(wds) * len(schedulers) * len(grad_norms))
        lr_idx = idx % len(lrs)
        wd_idx = (idx // len(lrs)) % len(wds)
        sched_idx = (idx // (len(lrs) * len(wds))) % len(schedulers)
        gn_idx = (idx // (len(lrs) * len(wds) * len(schedulers))) % len(grad_norms)
        
        lr = lrs[lr_idx]
        weight_decay = wds[wd_idx]
        use_lr_scheduler = schedulers[sched_idx]
        max_grad_norm = grad_norms[gn_idx]
    
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "use_lr_scheduler": use_lr_scheduler,
        "max_grad_norm": max_grad_norm,
    }


def run_sweep(
    list_lens: List[int],
    n_samples: int,
    output_csv: str,
    device: torch.device,
    strategy: str = "random",
    max_steps: int = 100000,
    batch_size: int = 2048,
):
    """Run hyperparameter sweep for specified list lengths"""
    
    fieldnames = [
        "list_len", "n_layers", "n_digits", "d_model", "n_heads",
        "lr", "weight_decay", "use_lr_scheduler", "warmup_steps", "max_grad_norm",
        "max_steps", "batch_size", "seed", "run_idx",
        "val_acc", "steps_trained", "train_time_sec",
        "params_total", "params_trainable", "val_examples"
    ]
    
    # Check if file exists
    file_exists = Path(output_csv).exists()
    
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for list_len in list_lens:
            n_layers = list_len  # Constraint: n_layers <= list_len (using equality)
            
            print(f"\n{'='*60}")
            print(f"Running sweep: list_len={list_len}, n_layers={n_layers}")
            print(f"{'='*60}")
            
            for run_idx in range(n_samples):
                # Sample hyperparameters
                hparams = sample_hyperparams(run_idx, n_samples, strategy)
                
                # Create config
                cfg = Config(
                    list_len=list_len,
                    n_digits=100,
                    d_model=64,
                    n_heads=1,
                    n_layers=n_layers,
                    lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"],
                    use_lr_scheduler=hparams["use_lr_scheduler"],
                    warmup_steps=1000,
                    max_grad_norm=hparams["max_grad_norm"],
                    max_steps=max_steps,
                    batch_size=batch_size,
                    seed=run_idx,
                )
                
                print(f"\n[{run_idx+1}/{n_samples}] lr={cfg.lr:.2e}, wd={cfg.weight_decay:.2e}, "
                      f"scheduler={cfg.use_lr_scheduler}, grad_clip={cfg.max_grad_norm}")
                
                # Train and evaluate
                metrics = train_and_evaluate(cfg, device)
                
                # Combine config and metrics for CSV
                row = {
                    "list_len": cfg.list_len,
                    "n_layers": cfg.n_layers,
                    "n_digits": cfg.n_digits,
                    "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads,
                    "lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "use_lr_scheduler": cfg.use_lr_scheduler,
                    "warmup_steps": cfg.warmup_steps,
                    "max_grad_norm": cfg.max_grad_norm,
                    "max_steps": cfg.max_steps,
                    "batch_size": cfg.batch_size,
                    "seed": cfg.seed,
                    "run_idx": run_idx,
                    **metrics
                }
                
                # Write row with file locking
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                writer.writerow(row)
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                print(f"  → Accuracy: {metrics['val_acc']:.4f} ({metrics['steps_trained']} steps, "
                      f"{metrics['train_time_sec']:.1f}s)")
                
                # Stop early if we hit 100%
                if metrics['val_acc'] >= 1.0:
                    print(f"  ✓ Achieved 100% accuracy!")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for 100% accuracy")
    parser.add_argument("--list-lens", type=int, nargs="+", default=[2, 3, 4],
                        help="List lengths to sweep over")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of hyperparameter samples per list_len")
    parser.add_argument("--output", type=str, default="hyperparam_sweep_results.csv",
                        help="Output CSV filename")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "grid"],
                        help="Sampling strategy (random or grid)")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Training batch size")
    
    args = parser.parse_args()
    
    device = pick_device(args.device)
    print(f"Using device: {device}")
    
    run_sweep(
        list_lens=args.list_lens,
        n_samples=args.n_samples,
        output_csv=args.output,
        device=device,
        strategy=args.strategy,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    
    print(f"\n{'='*60}")
    print(f"Sweep complete! Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
