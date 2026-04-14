# Design: 2-Layer Architecture Sweep

**Date:** 2026-04-14  
**Status:** Approved

## Overview

Grid search over architectural flags (ln, bias, wv, wo, mlp) and d_model scale for the 2-layer attention-only transformer on the list-copy task. Results are logged to wandb (`order-by-scale` project) and aggregated into a LaTeX table matching the existing report format.

## Scope

Two wandb sweeps, both in `project: order-by-scale`:

| Sweep | Config file | Parameters | Runs |
|---|---|---|---|
| Flag sweep | `sweeps/2layer_flags.yaml` | ln×bias×wv×wo×mlp (32 combos) × seeds 0–29, d_model=64 fixed | 960 |
| d_model sweep | `sweeps/2layer_dmodel.yaml` | d_model ∈ {8, 32, 128} × seeds 0–29, all flags False | 90 |

Total: **1050 runs**. d_model=64 FFFFF is intentionally only in the flag sweep (no double-counting).

## Files

```
sweeps/
  2layer_flags.yaml          # wandb sweep config: flag grid
  2layer_dmodel.yaml         # wandb sweep config: d_model grid
scripts/
  sweep_2layer.py            # sweep training function (called by both agents)
  make_2layer_table.py       # wandb API → LaTeX + CSV cache
slurm/
  submit_2layer_sweep.sh     # SLURM job: takes sweep_id as $1, runs wandb agent
models/
  2_layer_sweep/             # saved model checkpoints (FFFFF d_model=64 only)
results/
  2layer_sweep_cache.csv     # local cache written by make_2layer_table.py
```

## sweep_2layer.py

Follows the `sweep_sae.py` pattern. Entry point is `sweep_2layer()`:

1. `wandb.init(project="order-by-scale")` — sweep agent injects config
2. Read `wandb.config`: `d_model, use_ln, use_bias, use_wv, use_wo, use_mlp, seed`
3. `torch.manual_seed(seed)` + numpy/random seeds
4. `configure_runtime(list_len=2, seq_len=5, vocab=102, device=DEV, seed=seed)`
5. `make_model(...)`, `get_dataset(...)`, build DataLoaders
6. Train loop matching `train_model.py`: AdamW, eval every 100 steps, `max_steps=100000`, `early_stop_acc=0.999`. **No min_acc/retry logic** — with 30 seeds, a low-accuracy run is a data point, not a failure. Single training run per seed.
7. `wandb.log({"final/val_accuracy": best_acc})`
8. `wandb.summary["final/val_accuracy"] = best_acc` (for sweep UI sorting)
9. **Model saving:** if `d_model==64` and all flags False → save to `models/2_layer_sweep/<run_name>.pt`; all other configs → no model saved
10. `wandb.finish()`

Training hyperparameters fixed at `train_model.py` defaults: `lr=1e-3`, `weight_decay=0.01`, `train_batch_size=2048`, `val_batch_size=4096`.

**Dataset seeding:** always call `get_dataset(seed=0)` regardless of the run's training seed. `get_dataset` calls `torch.manual_seed(seed)` internally, so the train/val split (80/20 of 10,000 combinations) is identical across all seeds. The training seed only affects model weight initialisation and DataLoader shuffle order.

## Sweep YAMLs

### 2layer_flags.yaml
```yaml
program: scripts/sweep_2layer.py
method: grid
project: order-by-scale
parameters:
  use_ln:    {values: [true, false]}
  use_bias:  {values: [true, false]}
  use_wv:    {values: [true, false]}
  use_wo:    {values: [true, false]}
  use_mlp:   {values: [true, false]}
  seed:      {values: [0,1,2,...,29]}
  d_model:   {value: 64}
```

### 2layer_dmodel.yaml
```yaml
program: scripts/sweep_2layer.py
method: grid
project: order-by-scale
parameters:
  d_model:   {values: [8, 32, 128]}
  seed:      {values: [0,1,2,...,29]}
  use_ln:    {value: false}
  use_bias:  {value: false}
  use_wv:    {value: false}
  use_wo:    {value: false}
  use_mlp:   {value: false}
```

## make_2layer_table.py

CLI args: `--flags-sweep-ids` (one or more IDs, space-separated) and `--dmodel-sweep-ids`. Accepts multiple IDs per block so runs from additional sweeps are merged automatically.

Steps:
1. `api = wandb.Api(); runs = api.runs("order-by-scale", filters={"sweep": sweep_id})`
2. For each run extract config key `(d_model, ln, bias, wv, wo, mlp)` and `summary["final/val_accuracy"]`
3. Skip runs without a final accuracy (still in progress or crashed)
4. Group by config key → compute **mean, max, min, median, n_seeds** over all seeds
5. Save `results/2layer_sweep_cache.csv` (one row per run, for offline reuse)
6. Generate LaTeX table Block 1 (flag sweep, d_model=64) and Block 2 (d_model scale, all FFFFF)
7. Bold entries where mean ≥ 0.90 (matching existing table style)
8. For FFFFF d_model=64: print top-3 run names by `final/val_accuracy` → user manually keeps those files in `models/2_layer_sweep/`

Table columns match the provided example:
`d_model | LN | Bias | W_V | W_O | MLP | mean_acc | max_acc | min_acc | median_acc | n_seeds`

The LaTeX output also includes a compact version (mean only) matching the original table format exactly, for copy-paste into the report.

## SLURM

`submit_2layer_sweep.sh` accepts sweep ID as `$1`:

```bash
#SBATCH --job-name=2layer_sweep
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=28G

cd /home2/nchw73/Year4/L4_Project/list-comp-priv
uv sync && source .venv/bin/activate
wandb agent theo-farrell99-durham-university/order-by-scale/$1
```

Launch multiple agents: `sbatch submit_2layer_sweep.sh <sweep_id>` repeated N times.

## Incrementally adding seeds

To add seeds 30–59 later: create `2layer_flags_extra.yaml` with `seed: values: [30..59]`, run `wandb sweep` to get a new sweep ID, then pass both IDs to the table script:

```bash
python scripts/make_2layer_table.py \
  --flags-sweep-ids <original_id> <new_id> \
  --dmodel-sweep-ids <dmodel_id>
```

The script merges all runs from both IDs before computing per-config statistics.

## Out of scope

- Saving models for non-FFFFF configs
- Automatic deletion of non-top-3 FFFFF models (table script prints names; user decides)
- Sweeping over `n_layers`, `n_heads`, `n_digits`, or `list_len`
- Any modifications to `train_model.py`
