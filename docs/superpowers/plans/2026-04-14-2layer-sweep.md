# 2-Layer Architecture Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grid-search architectural flags (ln, bias, wv, wo, mlp) and d_model across 30 seeds using wandb sweeps, then generate a LaTeX results table.

**Architecture:** Two wandb sweep YAMLs drive a shared `sweep_2layer.py` training script (following the `sweep_sae.py` pattern). Results are aggregated offline by `make_2layer_table.py` via the wandb API. Model files are saved only for the FFFFF d_model=64 config.

**Tech Stack:** Python 3.12, PyTorch, wandb, pandas, `src.models`, `src.data.datasets`, SLURM

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `sweeps/2layer_flags.yaml` | Create | wandb grid: 32 flag combos × 30 seeds, d_model=64 |
| `sweeps/2layer_dmodel.yaml` | Create | wandb grid: d_model∈{8,32,128} × 30 seeds, all flags False |
| `scripts/sweep_2layer.py` | Create | Sweep training entry point (called by wandb agent) |
| `slurm/submit_2layer_sweep.sh` | Create | SLURM job: takes sweep_id as $1, runs wandb agent |
| `scripts/make_2layer_table.py` | Create | Queries wandb API → stats → LaTeX + CSV cache |
| `models/2_layer_sweep/` | Create (dir) | Saved FFFFF d_model=64 model checkpoints |

---

## Task 1: Sweep YAML configs

**Files:**
- Create: `sweeps/2layer_flags.yaml`
- Create: `sweeps/2layer_dmodel.yaml`

- [ ] **Step 1: Create the sweeps directory**

```bash
mkdir -p sweeps
```

- [ ] **Step 2: Write `sweeps/2layer_flags.yaml`**

```yaml
program: scripts/sweep_2layer.py
method: grid
project: order-by-scale
entity: theo-farrell99-durham-university
parameters:
  use_ln:
    values: [true, false]
  use_bias:
    values: [true, false]
  use_wv:
    values: [true, false]
  use_wo:
    values: [true, false]
  use_mlp:
    values: [true, false]
  seed:
    values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
  d_model:
    value: 64
```

- [ ] **Step 3: Write `sweeps/2layer_dmodel.yaml`**

```yaml
program: scripts/sweep_2layer.py
method: grid
project: order-by-scale
entity: theo-farrell99-durham-university
parameters:
  d_model:
    values: [8, 32, 128]
  seed:
    values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
  use_ln:
    value: false
  use_bias:
    value: false
  use_wv:
    value: false
  use_wo:
    value: false
  use_mlp:
    value: false
```

- [ ] **Step 4: Verify YAML syntax**

```bash
source .venv/bin/activate
python -c "
import yaml
for f in ['sweeps/2layer_flags.yaml', 'sweeps/2layer_dmodel.yaml']:
    with open(f) as fh:
        cfg = yaml.safe_load(fh)
    n = 1
    for p in cfg['parameters'].values():
        if 'values' in p:
            n *= len(p['values'])
    print(f'{f}: {n} runs, method={cfg[\"method\"]}')
"
```

Expected output:
```
sweeps/2layer_flags.yaml: 960 runs, method=grid
sweeps/2layer_dmodel.yaml: 90 runs, method=grid
```

- [ ] **Step 5: Commit**

```bash
rtk git add sweeps/
rtk git commit -m "feat: add wandb sweep configs for 2-layer flag and d_model grids"
```

---

## Task 2: `scripts/sweep_2layer.py`

**Files:**
- Create: `scripts/sweep_2layer.py`

This follows the `sweep_sae.py` pattern exactly. Key ordering constraint: `get_dataset(seed=0)` calls `torch.manual_seed(0)` internally — always call `set_seeds(training_seed)` **after** the dataset call to reset the RNG for model initialisation.

- [ ] **Step 1: Write the file**

```python
"""
sweep_2layer.py

W&B sweep entry point for the 2-layer architecture grid search.
Covers:
  - Flag sweep (sweep_2layer_flags.yaml): 32 flag combos x 30 seeds, d_model=64
  - d_model sweep (sweep_2layer_dmodel.yaml): d_model in {8,32,128} x 30 seeds, all flags False

Called directly by wandb agent; config is injected via wandb.init().
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
from dotenv import load_dotenv
import wandb

from src.utils.runtime import configure_runtime
from src.models.transformer import make_model
from src.models.utils import save_model, accuracy
from src.data.datasets import get_dataset

WANDB_PROJECT = "order-by-scale"
LIST_LEN = 2
N_DIGITS = 100
VOCAB = N_DIGITS + 2       # 102
SEQ_LEN = LIST_LEN * 2 + 1  # 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed training hyperparameters (match train_model.py defaults)
LR = 1e-3
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 2048
VAL_BATCH_SIZE = 4096
MAX_STEPS = 100_000
EARLY_STOP_ACC = 0.999


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_dl, val_dl, max_steps: int, early_stop_acc: float) -> float:
    """Train for up to max_steps, return best val accuracy seen."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = torch.nn.CrossEntropyLoss()
    data_iter = itertools.cycle(train_dl)
    best_acc = 0.0

    pbar = tqdm(range(max_steps), desc="Training", leave=False)
    for step in pbar:
        inputs, targets = next(data_iter)
        logits = model(inputs.to(DEV))[:, LIST_LEN + 1:].reshape(-1, VOCAB)
        loss = ce(logits, targets[:, LIST_LEN + 1:].reshape(-1).to(DEV))
        loss.backward()
        opt.step()
        opt.zero_grad()

        if (step + 1) % 100 == 0:
            acc = accuracy(model, val_dl)
            wandb.log({"val/accuracy": acc, "train/loss": loss.item(), "step": step + 1})
            if acc > best_acc:
                best_acc = acc
            pbar.set_postfix({"acc": f"{acc:.4f}", "best": f"{best_acc:.4f}"})
            if acc >= early_stop_acc:
                print(f"Early stop at step {step + 1}: acc={acc:.4f}")
                break

    return best_acc


def sweep_2layer() -> None:
    load_dotenv()
    run = wandb.init(project=WANDB_PROJECT)
    config = wandb.config

    d_model  = int(config.d_model)
    use_ln   = bool(config.use_ln)
    use_bias = bool(config.use_bias)
    use_wv   = bool(config.use_wv)
    use_wo   = bool(config.use_wo)
    use_mlp  = bool(config.use_mlp)
    seed     = int(config.seed)

    def tf(b):
        return "T" if b else "F"

    run.name = (
        f"d{d_model}_"
        f"ln{tf(use_ln)}_bias{tf(use_bias)}_"
        f"wv{tf(use_wv)}_wo{tf(use_wo)}_mlp{tf(use_mlp)}_"
        f"s{seed}"
    )

    print(f"\n{'='*60}")
    print(f"Run: {run.name}")
    print(f"{'='*60}\n")

    # --- Dataset (always seed=0 so the train/val split is identical across runs) ---
    # NOTE: get_dataset calls torch.manual_seed(seed) internally, so we MUST
    # call set_seeds(training_seed) AFTER this to reset the RNG for model init.
    configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV, seed=0)
    train_ds, val_ds = get_dataset(
        list_len=LIST_LEN,
        n_digits=N_DIGITS,
        train_split=0.8,
        mask_tok=N_DIGITS,
        sep_tok=N_DIGITS + 1,
        seed=0,
    )
    train_dl = DataLoader(train_ds, min(TRAIN_BATCH_SIZE, len(train_ds)),
                          shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   min(VAL_BATCH_SIZE,   len(val_ds)),
                          drop_last=False)

    # --- Re-seed for model init (training seed only) ---
    set_seeds(seed)

    # --- Model ---
    model = make_model(
        n_layers=2,
        n_heads=1,
        d_model=d_model,
        ln=use_ln,
        use_bias=use_bias,
        use_wv=use_wv,
        use_wo=use_wo,
        attn_only=not use_mlp,
    ).to(DEV)

    # --- Train ---
    best_acc = train(model, train_dl, val_dl, MAX_STEPS, EARLY_STOP_ACC)

    # --- Log final accuracy ---
    wandb.log({"final/val_accuracy": best_acc})
    wandb.summary["final/val_accuracy"] = best_acc
    print(f"Final val accuracy: {best_acc:.4f}")

    # --- Save model for FFFFF d_model=64 only ---
    is_fffff = not any([use_ln, use_bias, use_wv, use_wo, use_mlp])
    if is_fffff and d_model == 64:
        save_dir = Path(__file__).parent.parent / "models" / "2_layer_sweep"
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{run.name}.pt"
        save_model(model, str(model_path))

    wandb.finish()


if __name__ == "__main__":
    sweep_2layer()
```

- [ ] **Step 2: Smoke-test imports and dataset loading (offline, no GPU needed)**

```bash
source .venv/bin/activate
WANDB_MODE=offline python -c "
import sys; sys.path.insert(0, '.')
from src.utils.runtime import configure_runtime
from src.models.transformer import make_model
from src.models.utils import accuracy, save_model
from src.data.datasets import get_dataset
from torch.utils.data import DataLoader

configure_runtime(list_len=2, seq_len=5, vocab=102, device='cpu', seed=0)
train_ds, val_ds = get_dataset(list_len=2, n_digits=100, train_split=0.8,
                                mask_tok=100, sep_tok=101, seed=0)
train_dl = DataLoader(train_ds, 256, shuffle=True, drop_last=True)
val_dl   = DataLoader(val_ds,   256, drop_last=False)
model = make_model(n_layers=2, n_heads=1, d_model=64,
                   ln=False, use_bias=False, use_wv=False, use_wo=False,
                   attn_only=True).to('cpu')
acc = accuracy(model, val_dl)
print(f'Dataset OK: train={len(train_ds)}, val={len(val_ds)}')
print(f'Random-init accuracy (should be ~0.01): {acc:.4f}')
"
```

Expected:
```
Dataset OK: train=8000, val=2000
Random-init accuracy (should be ~0.01): 0.01xx
```

- [ ] **Step 3: Smoke-test a short training run (100 steps, offline wandb)**

```bash
source .venv/bin/activate
WANDB_MODE=offline python -c "
import sys, os, random, itertools
sys.path.insert(0, '.')
import numpy as np, torch
from torch.utils.data import DataLoader
import wandb
from src.utils.runtime import configure_runtime
from src.models.transformer import make_model
from src.models.utils import accuracy
from src.data.datasets import get_dataset

wandb.init(project='order-by-scale', mode='offline', config={
    'd_model': 64, 'use_ln': False, 'use_bias': False,
    'use_wv': False, 'use_wo': False, 'use_mlp': False, 'seed': 0
})
configure_runtime(list_len=2, seq_len=5, vocab=102, device='cpu', seed=0)
train_ds, val_ds = get_dataset(list_len=2, n_digits=100, train_split=0.8,
                                mask_tok=100, sep_tok=101, seed=0)
train_dl = DataLoader(train_ds, 256, shuffle=True, drop_last=True)
val_dl   = DataLoader(val_ds,   256, drop_last=False)
torch.manual_seed(0)
model = make_model(n_layers=2, n_heads=1, d_model=64, ln=False,
                   use_bias=False, use_wv=False, use_wo=False, attn_only=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
ce = torch.nn.CrossEntropyLoss()
data_iter = itertools.cycle(train_dl)
for step in range(100):
    inputs, targets = next(data_iter)
    logits = model(inputs)[:, 3:].reshape(-1, 102)
    loss = ce(logits, targets[:, 3:].reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
acc = accuracy(model, val_dl)
print(f'100-step acc: {acc:.4f}  loss: {loss.item():.4f}')
wandb.finish()
print('Smoke test PASSED')
"
```

Expected: prints a loss and accuracy, ends with `Smoke test PASSED` (no exceptions).

- [ ] **Step 4: Commit**

```bash
rtk git add scripts/sweep_2layer.py
rtk git commit -m "feat: add sweep_2layer.py training entry point for wandb agent"
```

---

## Task 3: `slurm/submit_2layer_sweep.sh`

**Files:**
- Create: `slurm/submit_2layer_sweep.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
#SBATCH --job-name=2layer_sweep
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=28G

# Usage: sbatch slurm/submit_2layer_sweep.sh <sweep_id>
# Launch multiple times for parallel agents, e.g.:
#   for i in {1..8}; do sbatch slurm/submit_2layer_sweep.sh <sweep_id>; done

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: No sweep ID provided. Usage: sbatch submit_2layer_sweep.sh <sweep_id>"
    exit 1
fi

cd /home2/nchw73/Year4/L4_Project/list-comp-priv
uv sync
source .venv/bin/activate

echo "[slurm] Job running on node: $(hostname)"
echo "[slurm] Sweep ID: $SWEEP_ID"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'[slurm] CUDA Available: {torch.cuda.is_available()}'); print(f'[slurm] Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

wandb agent theo-farrell99-durham-university/order-by-scale/$SWEEP_ID

echo "[slurm] Finished at $(date)"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x slurm/submit_2layer_sweep.sh
```

- [ ] **Step 3: Verify the script is valid bash**

```bash
bash -n slurm/submit_2layer_sweep.sh
echo "Syntax OK: $?"
```

Expected: `Syntax OK: 0`

- [ ] **Step 4: Commit**

```bash
rtk git add slurm/submit_2layer_sweep.sh
rtk git commit -m "feat: add SLURM submit script for 2-layer sweep agents"
```

---

## Task 4: `scripts/make_2layer_table.py`

**Files:**
- Create: `scripts/make_2layer_table.py`

Queries wandb API for all runs in the given sweep IDs, groups by config key, computes per-config statistics, writes a local CSV cache, and prints two LaTeX table blocks.

- [ ] **Step 1: Write the file**

```python
"""
make_2layer_table.py

Query wandb API for 2-layer sweep results and produce:
  1. results/2layer_sweep_cache.csv   — one row per completed run (local cache)
  2. LaTeX Block 1: flag sweep (d_model=64, all 32 flag combos)
  3. LaTeX Block 2: d_model sweep (FFFFF, d_model in {8,32,64,128})
  4. Compact LaTeX table matching original report format (mean only)
  5. Top-3 FFFFF d_model=64 model filenames to keep

Usage:
    python scripts/make_2layer_table.py \\
        --flags-sweep-ids <id1> [<id2> ...] \\
        --dmodel-sweep-ids <id1> [<id2> ...]

Add more sweep IDs later (e.g. for extra seeds) by passing multiple IDs;
the script merges all matching runs before computing statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import List
import pandas as pd
import numpy as np
import wandb

WANDB_ENTITY  = "theo-farrell99-durham-university"
WANDB_PROJECT = "order-by-scale"
CACHE_PATH    = Path(__file__).parent.parent / "results" / "2layer_sweep_cache.csv"
BOLD_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# wandb fetch
# ---------------------------------------------------------------------------

def fetch_runs(sweep_ids: List[str]) -> pd.DataFrame:
    """Pull all completed runs from the given sweep IDs and return a DataFrame."""
    api = wandb.Api()
    rows = []
    for sweep_id in sweep_ids:
        print(f"Fetching sweep {sweep_id} ...")
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"sweep": sweep_id},
        )
        for run in runs:
            if "final/val_accuracy" not in run.summary:
                continue  # skip in-progress or crashed runs
            rows.append({
                "sweep_id":   sweep_id,
                "run_name":   run.name,
                "run_id":     run.id,
                "d_model":    int(run.config.get("d_model", 64)),
                "use_ln":     bool(run.config.get("use_ln", False)),
                "use_bias":   bool(run.config.get("use_bias", False)),
                "use_wv":     bool(run.config.get("use_wv", False)),
                "use_wo":     bool(run.config.get("use_wo", False)),
                "use_mlp":    bool(run.config.get("use_mlp", False)),
                "seed":       int(run.config.get("seed", -1)),
                "val_accuracy": float(run.summary["final/val_accuracy"]),
            })
    if not rows:
        raise SystemExit("No completed runs found — check sweep IDs and that runs have logged final/val_accuracy.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

CONFIG_KEYS = ["d_model", "use_ln", "use_bias", "use_wv", "use_wo", "use_mlp"]


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Group by config key, return per-config mean/max/min/median/n_seeds."""
    stats = (
        df.groupby(CONFIG_KEYS)["val_accuracy"]
        .agg(mean="mean", max="max", min="min", median="median", n_seeds="count")
        .reset_index()
    )
    return stats


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def tf(b: bool) -> str:
    return "T" if b else "F"


def fmt_acc(val: float, bold: bool = False) -> str:
    s = f"{val:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def make_flags_table(stats: pd.DataFrame) -> str:
    """Block 1: flag sweep rows (d_model=64), all 32 configs."""
    block = stats[stats["d_model"] == 64].sort_values(
        ["use_ln", "use_bias", "use_wv", "use_wo", "use_mlp"]
    )
    lines = [
        r"\begin{tabular}{ccccccccccc}",
        r"\toprule",
        r"$d_{\text{model}}$ & LN & Bias & $W_V$ & $W_O$ & MLP"
        r" & Mean & Max & Min & Median & $n$ \\",
        r"\midrule",
    ]
    for _, row in block.iterrows():
        bold = row["mean"] >= BOLD_THRESHOLD
        lines.append(
            f"64 & {tf(row['use_ln'])} & {tf(row['use_bias'])} & "
            f"{tf(row['use_wv'])} & {tf(row['use_wo'])} & {tf(row['use_mlp'])} & "
            f"{fmt_acc(row['mean'], bold)} & "
            f"{row['max']:.4f} & {row['min']:.4f} & {row['median']:.4f} & "
            f"{int(row['n_seeds'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_dmodel_table(stats: pd.DataFrame) -> str:
    """Block 2: d_model sweep rows (all flags False), sorted descending."""
    is_fffff = (
        ~stats["use_ln"] & ~stats["use_bias"] &
        ~stats["use_wv"] & ~stats["use_wo"] & ~stats["use_mlp"]
    )
    block = stats[is_fffff].sort_values("d_model", ascending=False)
    lines = [
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"$d_{\text{model}}$ & Mean & Max & Min & Median & $n$ \\",
        r"\midrule",
    ]
    for _, row in block.iterrows():
        bold = row["mean"] >= BOLD_THRESHOLD
        lines.append(
            f"{int(row['d_model'])} & "
            f"{fmt_acc(row['mean'], bold)} & "
            f"{row['max']:.4f} & {row['min']:.4f} & {row['median']:.4f} & "
            f"{int(row['n_seeds'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_compact_table(stats: pd.DataFrame) -> str:
    """Compact mean-only table matching original report format (both blocks)."""
    # Block 1: flag sweep (d_model=64)
    b1 = stats[stats["d_model"] == 64].sort_values(
        ["use_ln", "use_bias", "use_wv", "use_wo", "use_mlp"]
    )
    # Block 2: FFFFF across d_models
    is_fffff = (
        ~stats["use_ln"] & ~stats["use_bias"] &
        ~stats["use_wv"] & ~stats["use_wo"] & ~stats["use_mlp"]
    )
    b2 = stats[is_fffff].sort_values("d_model", ascending=False)

    lines = [
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"$d_{\text{model}}$ & LN & Bias & $W_V$ & $W_O$ & MLP & Accuracy \\",
        r"\midrule",
    ]
    for _, row in b1.iterrows():
        bold = row["mean"] >= BOLD_THRESHOLD
        lines.append(
            f"64 & {tf(row['use_ln'])} & {tf(row['use_bias'])} & "
            f"{tf(row['use_wv'])} & {tf(row['use_wo'])} & {tf(row['use_mlp'])} & "
            f"{fmt_acc(row['mean'], bold)} \\\\"
        )
    lines.append(r"\midrule")
    for _, row in b2.iterrows():
        bold = row["mean"] >= BOLD_THRESHOLD
        lines.append(
            f"{int(row['d_model'])} & F & F & F & F & F & "
            f"{fmt_acc(row['mean'], bold)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def print_top3_fffff(df: pd.DataFrame) -> None:
    """Print the top-3 FFFFF d_model=64 run names to keep from models/2_layer_sweep/."""
    is_fffff_64 = (
        (df["d_model"] == 64) &
        ~df["use_ln"] & ~df["use_bias"] &
        ~df["use_wv"] & ~df["use_wo"] & ~df["use_mlp"]
    )
    top3 = df[is_fffff_64].nlargest(3, "val_accuracy")
    print("\nTop-3 FFFFF d_model=64 models to keep in models/2_layer_sweep/:")
    for _, row in top3.iterrows():
        print(f"  {row['run_name']}.pt   (seed={row['seed']}, acc={row['val_accuracy']:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 2-layer sweep results table from wandb")
    p.add_argument("--flags-sweep-ids",  nargs="+", required=True,
                   metavar="ID", help="Sweep ID(s) for the flag grid (d_model=64)")
    p.add_argument("--dmodel-sweep-ids", nargs="+", required=True,
                   metavar="ID", help="Sweep ID(s) for the d_model grid (FFFFF)")
    p.add_argument("--no-cache", action="store_true",
                   help="Re-fetch from wandb even if cache exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_sweep_ids = args.flags_sweep_ids + args.dmodel_sweep_ids

    # Fetch or load cache
    if CACHE_PATH.exists() and not args.no_cache:
        print(f"Loading cached runs from {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH)
        # Only re-fetch sweeps not already in cache
        cached_ids = set(df["sweep_id"].unique())
        missing = [sid for sid in all_sweep_ids if sid not in cached_ids]
        if missing:
            print(f"Fetching {len(missing)} new sweep(s) ...")
            new_df = fetch_runs(missing)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(CACHE_PATH, index=False)
            print(f"Cache updated: {len(df)} total runs")
    else:
        df = fetch_runs(all_sweep_ids)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        print(f"Cache written: {CACHE_PATH}  ({len(df)} runs)")

    # Filter to requested sweeps only (cache may have others)
    df = df[df["sweep_id"].isin(all_sweep_ids)].copy()
    print(f"\nRuns in requested sweeps: {len(df)}")

    # Compute per-config statistics
    stats = compute_stats(df)
    print(f"Unique configs: {len(stats)}\n")

    # Print tables
    print("=" * 70)
    print("BLOCK 1 — Flag sweep (d_model=64, all 32 configs):")
    print("=" * 70)
    print(make_flags_table(stats))

    print("\n" + "=" * 70)
    print("BLOCK 2 — d_model sweep (all flags False):")
    print("=" * 70)
    print(make_dmodel_table(stats))

    print("\n" + "=" * 70)
    print("COMPACT TABLE (mean only, report format):")
    print("=" * 70)
    print(make_compact_table(stats))

    print_top3_fffff(df)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the table generation logic with mock data (no wandb needed)**

```bash
source .venv/bin/activate
python -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np

# Import the helper functions
sys.argv = ['make_2layer_table.py',
            '--flags-sweep-ids', 'fake1',
            '--dmodel-sweep-ids', 'fake2']

# Build mock DataFrame matching the schema
import itertools
rows = []
for ln, bias, wv, wo, mlp in itertools.product([True, False], repeat=5):
    for seed in range(3):
        rows.append({
            'sweep_id': 'fake1', 'run_name': f'run_{seed}',
            'run_id': f'id_{seed}', 'd_model': 64,
            'use_ln': ln, 'use_bias': bias, 'use_wv': wv,
            'use_wo': wo, 'use_mlp': mlp, 'seed': seed,
            'val_accuracy': np.random.uniform(0.5, 1.0),
        })
for d in [8, 32, 128]:
    for seed in range(3):
        rows.append({
            'sweep_id': 'fake2', 'run_name': f'dmodel_{d}_s{seed}',
            'run_id': f'did_{seed}', 'd_model': d,
            'use_ln': False, 'use_bias': False, 'use_wv': False,
            'use_wo': False, 'use_mlp': False, 'seed': seed,
            'val_accuracy': np.random.uniform(0.3, 0.9),
        })
df = pd.DataFrame(rows)

from scripts.make_2layer_table import compute_stats, make_flags_table, make_dmodel_table, make_compact_table, print_top3_fffff
stats = compute_stats(df)
print('compute_stats OK:', len(stats), 'configs')
t1 = make_flags_table(stats)
assert r'\begin{tabular}' in t1 and r'\end{tabular}' in t1
print('make_flags_table OK')
t2 = make_dmodel_table(stats)
assert r'\begin{tabular}' in t2
print('make_dmodel_table OK')
t3 = make_compact_table(stats)
assert r'\midrule' in t3
print('make_compact_table OK')
print_top3_fffff(df)
print('All table helpers PASSED')
"
```

Expected: prints `compute_stats OK: 32 configs`, all OK messages, and three model names.

- [ ] **Step 3: Commit**

```bash
rtk git add scripts/make_2layer_table.py
rtk git commit -m "feat: add make_2layer_table.py to query wandb and generate LaTeX results table"
```

---

## Task 5: Register sweeps and verify end-to-end (1 trial run each)

This task is run on the HPC (internet access needed for wandb). Do not run locally.

- [ ] **Step 1: Register both sweeps with wandb**

```bash
source .venv/bin/activate
wandb sweep sweeps/2layer_flags.yaml
# Note the sweep ID printed, e.g. abc12345
wandb sweep sweeps/2layer_dmodel.yaml
# Note the sweep ID printed, e.g. def67890
```

- [ ] **Step 2: Run 1 trial agent for each sweep to confirm no errors**

```bash
source .venv/bin/activate
# Flags sweep - run 1 trial
WANDB_AGENT_MAX_INITIAL_FAILURES=1 wandb agent \
    theo-farrell99-durham-university/order-by-scale/<flags_sweep_id> \
    --count 1

# d_model sweep - run 1 trial
wandb agent \
    theo-farrell99-durham-university/order-by-scale/<dmodel_sweep_id> \
    --count 1
```

Expected: a wandb run completes, logs `final/val_accuracy`, and for the FFFFF run a file appears in `models/2_layer_sweep/`.

- [ ] **Step 3: Verify model was saved for FFFFF trial**

```bash
ls models/2_layer_sweep/
```

Expected: one `.pt` file named like `d64_lnF_biasF_wvF_woF_mlpF_s<N>.pt`.

- [ ] **Step 4: Verify table script runs against the 1-run data**

```bash
source .venv/bin/activate
python scripts/make_2layer_table.py \
    --flags-sweep-ids <flags_sweep_id> \
    --dmodel-sweep-ids <dmodel_sweep_id>
```

Expected: prints partial tables (1 row each) without errors.

- [ ] **Step 5: Launch full parallel agents**

```bash
# Flags sweep: 8 parallel agents (each picks up ~120 runs)
for i in {1..8}; do
    sbatch slurm/submit_2layer_sweep.sh <flags_sweep_id>
done

# d_model sweep: 4 parallel agents (each picks up ~23 runs)
for i in {1..4}; do
    sbatch slurm/submit_2layer_sweep.sh <dmodel_sweep_id>
done
```

- [ ] **Step 6: Commit any changes from trial run, update EXPERIMENTS.md**

Append to `EXPERIMENTS.md`:

```markdown
## 2-Layer Architecture Sweep (2026-04-14)

**Command:**
```bash
wandb sweep sweeps/2layer_flags.yaml   # → <flags_sweep_id>
wandb sweep sweeps/2layer_dmodel.yaml  # → <dmodel_sweep_id>
for i in {1..8}; do sbatch slurm/submit_2layer_sweep.sh <flags_sweep_id>; done
for i in {1..4}; do sbatch slurm/submit_2layer_sweep.sh <dmodel_sweep_id>; done
```

**Output paths:**
- Sweep runs: wandb project `order-by-scale`
- Models (FFFFF d_model=64 only): `models/2_layer_sweep/`
- Results cache: `results/2layer_sweep_cache.csv`

**Config:** 32 flag combos × 30 seeds (d_model=64) + d_model∈{8,32,128} × 30 seeds (FFFFF) = 1050 runs total

**Table generation:**
```bash
python scripts/make_2layer_table.py \
    --flags-sweep-ids <flags_sweep_id> \
    --dmodel-sweep-ids <dmodel_sweep_id>
```
```

```bash
rtk git add EXPERIMENTS.md
rtk git commit -m "docs: record 2-layer sweep experiment in EXPERIMENTS.md"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `sweeps/2layer_flags.yaml`: 32 combos × 30 seeds | Task 1 |
| `sweeps/2layer_dmodel.yaml`: {8,32,128} × 30 seeds | Task 1 |
| `sweep_2layer.py` follows sweep_sae.py pattern | Task 2 |
| Fixed dataset seed=0 across all runs | Task 2 (seed ordering documented) |
| No min_acc/retry logic | Task 2 (single train() call) |
| Log `final/val_accuracy` to wandb summary | Task 2 |
| Save FFFFF d_model=64 models to `models/2_layer_sweep/` | Task 2 |
| `submit_2layer_sweep.sh` takes $1 sweep ID | Task 3 |
| `make_2layer_table.py`: multiple sweep IDs per block | Task 4 |
| Per-config mean/max/min/median/n_seeds | Task 4 |
| Local CSV cache | Task 4 |
| Two LaTeX blocks + compact format | Task 4 |
| Top-3 FFFFF model names printed | Task 4 |
| EXPERIMENTS.md entry | Task 5 |
