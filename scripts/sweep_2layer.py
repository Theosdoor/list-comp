"""
W&B Sweep Script for 2-Layer Transformer Architecture Sweep

Grid search over architectural flags (ln, bias, wv, wo, mlp) and d_model scale.
Results are logged to wandb project "order-by-scale".

Speedups vs plain train_model.py:
  - torch.autocast (float16) + GradScaler on CUDA
  - torch.backends.cudnn.benchmark for fixed input shapes
  - pin_memory + non_blocking transfers
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import random
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from dotenv import load_dotenv

from src.utils.runtime import configure_runtime
from src.models.transformer import make_model
from src.models.utils import accuracy
from src.data.datasets import get_dataset


LIST_LEN = 2
SEQ_LEN  = LIST_LEN * 2 + 1  # 5
N_DIGITS = 100
VOCAB    = N_DIGITS + 2       # 102

SAVE_DIR = "models/2_layer_sweep"

DEV     = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEV == "cuda")

# cuDNN auto-tunes conv/GEMM kernels once per fixed input shape — free win.
if DEV == "cuda":
    torch.backends.cudnn.benchmark = True


def sweep_2layer():
    load_dotenv()
    run = wandb.init(project="order-by-scale")
    cfg = wandb.config

    d_model  = cfg.d_model
    use_ln   = bool(cfg.use_ln)
    use_bias = bool(cfg.use_bias)
    use_wv   = bool(cfg.use_wv)
    use_wo   = bool(cfg.use_wo)
    use_mlp  = bool(cfg.use_mlp)
    seed     = int(cfg.seed)

    flags_str = "".join(
        "T" if v else "F"
        for v in [use_ln, use_bias, use_wv, use_wo, use_mlp]
    )
    run.name = f"L2_D{d_model}_{flags_str}_s{seed}"

    # ---- Dataset (fixed split, seed=0 so train/val is identical across all runs) ----
    # Must happen BEFORE setting the training seed because get_dataset calls
    # torch.manual_seed(0) internally and would overwrite our seed.
    train_ds, val_ds = get_dataset(
        list_len=LIST_LEN,
        n_digits=N_DIGITS,
        train_split=0.8,
        mask_tok=N_DIGITS,
        sep_tok=N_DIGITS + 1,
        seed=0,
    )

    # ---- Training seed (model weight init + DataLoader shuffle order) ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # configure_runtime also calls torch.manual_seed(seed), reinforcing the above.
    configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV, seed=seed)

    train_dl = DataLoader(
        train_ds, batch_size=2048, shuffle=True, drop_last=True,
        pin_memory=USE_AMP,
    )
    val_dl = DataLoader(
        val_ds, batch_size=4096, drop_last=False,
        pin_memory=USE_AMP,
    )

    model = make_model(
        n_layers=2,
        n_heads=1,
        d_model=d_model,
        ln=use_ln,
        use_bias=use_bias,
        use_wv=use_wv,
        use_wo=use_wo,
        attn_only=not use_mlp,
    )

    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    ce     = torch.nn.CrossEntropyLoss()

    dl_cycle       = itertools.cycle(train_dl)
    max_steps      = 100_000
    early_stop_acc = 0.999
    best_acc       = 0.0

    for step in range(max_steps):
        model.train()
        inputs, targets = next(dl_cycle)
        inputs  = inputs.to(DEV, non_blocking=True)
        targets = targets.to(DEV, non_blocking=True)

        with torch.autocast(device_type=DEV, dtype=torch.float16, enabled=USE_AMP):
            logits = model(inputs)[:, LIST_LEN + 1:].reshape(-1, VOCAB)
            loss   = ce(logits, targets[:, LIST_LEN + 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

        if (step + 1) % 100 == 0:
            acc      = accuracy(model, val_dl)
            best_acc = max(best_acc, acc)
            wandb.log({
                "train/loss":   loss.item(),
                "val/accuracy": acc,
                "step":         step + 1,
            })
            if acc >= early_stop_acc:
                break

    # Final eval (covers the case where last logged eval was >100 steps ago)
    final_acc = accuracy(model, val_dl)
    best_acc  = max(best_acc, final_acc)

    wandb.log({"final/val_accuracy": best_acc})
    wandb.summary["final/val_accuracy"] = best_acc

    # Save FFFFF d_model=64 baseline models only
    if d_model == 64 and not any([use_ln, use_bias, use_wv, use_wo, use_mlp]):
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f"{run.name}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    wandb.finish()


if __name__ == "__main__":
    sweep_2layer()
