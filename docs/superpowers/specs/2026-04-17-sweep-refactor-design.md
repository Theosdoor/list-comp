# Sweep Refactor & Listlen×Nlayers Sweep Design

**Date:** 2026-04-17  
**Status:** Approved

## Overview

Consolidate the old `results/grid_search/` scripts into the modern W&B sweep infrastructure, generalise `scripts/sweep_2layer.py` into `scripts/sweep_transformer.py`, and add a new `listlen×nlayers` sweep to the `order-by-scale` W&B project.

After this work the project has four canonical sweeps, all in `sweeps/` and all running against `scripts/sweep_transformer.py`:

| YAML | Axis | Runs |
|---|---|---|
| `2layer_flags.yaml` | `{LN, BIAS, WV, WO, MLP}` × 30 seeds | 960 |
| `2layer_dmodel.yaml` | `d_model ∈ [8,32,128,256]` × 30 seeds | 120 |
| `2layer_nheads.yaml` | `n_heads ∈ [1,2,4,8]` × 30 seeds | 120 |
| `listlen_nlayers.yaml` | `list_len ∈ [1..10]` × `n_layers ∈ [1..10]` × 30 seeds | 3000 |

---

## 1. `src/models/utils.py` — add `count_params`

Add one new exported function:

```python
def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
```

Add `"count_params"` to `__all__`. No other changes to this file.

---

## 2. `scripts/sweep_transformer.py` — generalised sweep script

Derived from `scripts/sweep_2layer.py` with the following changes only:

**Parameterise `list_len` and `n_layers` from W&B config:**
```python
list_len = getattr(config, "list_len", 2)
n_layers  = getattr(config, "n_layers", 2)
```
`SEQ_LEN = list_len * 2 + 1` and `VOCAB = N_DIGITS + 2` are computed from these. The constants `LIST_LEN = 2` and `N_LAYERS = 2` at module level are removed.

**Log parameter counts:**
```python
from src.models.utils import count_params
...
total_params, trainable_params = count_params(model)
wandb.log({"model/params_total": total_params, "model/params_trainable": trainable_params})
```

**Model saving condition — unchanged from `sweep_2layer.py`:**
```python
if (not use_ln and not use_bias and not use_wv and not use_wo and not use_mlp and d_model == 64):
    ...
```
Note: this condition will trigger for every run in the `listlen_nlayers` sweep (all flags false, `d_model=64`), saving models for all 100 arch combos × 30 seeds to `models/2_layer_sweep/`. Intentional.

**Run name** extended to include `list_len` and `n_layers`:
```
d{d_model}_L{list_len}_N{n_layers}_h{n_heads}_ln{T/F}_...
```

Everything else (training loop, DataLoaders, W&B init, early stopping) is identical to `sweep_2layer.py`.

---

## 3. `scripts/sweep_2layer.py` — left untouched

`sweep_2layer.py` is not modified. The currently-running `2layer_dmodel` sweep is registered against it and changing it mid-run is unnecessary risk. `sweep_transformer.py` is a fully independent new file. Once the running sweep finishes, future sweeps can be created from updated YAMLs pointing to `sweep_transformer.py`.

---

## 4. `sweeps/listlen_nlayers.yaml` — new sweep

```yaml
program: scripts/sweep_transformer.py
method: grid
project: order-by-scale
entity: theo-farrell99-durham-university
parameters:
  list_len:
    values: [1,2,3,4,5,6,7,8,9,10]
  n_layers:
    values: [1,2,3,4,5,6,7,8,9,10]
  seed:
    values: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
  d_model:
    value: 64
  n_heads:
    value: 1
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

Run with:
```bash
wandb sweep sweeps/listlen_nlayers.yaml
wandb agent <sweep_id>   # or sbatch slurm/submit_sae_sweep.sh <sweep_id>
```

---

## 5. Update existing sweep YAMLs

Change `program:` in all three existing YAMLs:
```
- program: scripts/sweep_2layer.py
+ program: scripts/sweep_transformer.py
```

Files: `sweeps/2layer_flags.yaml`, `sweeps/2layer_dmodel.yaml`, `sweeps/2layer_nheads.yaml`.

This only affects **new sweeps** created from these YAMLs. The currently-running `2layer_dmodel` sweep is unaffected (it uses a W&B-registered sweep ID, not the local file).

---

## 6. Delete legacy files

- `results/grid_search/gridsearch_utils.py`
- `results/grid_search/listlen_grid_search.py`

The rest of `results/grid_search/` (`nb_results_heatmap.py`, `results.csv`, `ablation_res.csv`) is kept as historical reference.

---

## Data Flow

```
sweeps/listlen_nlayers.yaml
        │  wandb sweep
        ▼
W&B sweep controller (order-by-scale project)
        │  wandb agent <id>  [N parallel SLURM workers]
        ▼
scripts/sweep_transformer.py
  ├── configure_runtime(list_len, ...)
  ├── get_dataset(list_len, n_digits, ...)
  ├── make_model(n_layers, n_heads, d_model, ...)
  ├── src.models.utils.count_params(model) → wandb.log
  ├── src.models.train.train(...)  → wandb.log (per step)
  └── wandb.summary[final/val_accuracy]
```

---

## What is NOT changing

- `src/models/train.py` — training loop unchanged
- `src/data/datasets.py` — dataset generation unchanged
- `src/utils/runtime.py` — runtime config unchanged
- All other sweep YAMLs (`btksae_sweep.yaml`, `jumprelu_sweep.yaml`, `matryoshka_sweep.yaml`)
- SLURM submit scripts
