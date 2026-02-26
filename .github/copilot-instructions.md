# Copilot Instructions

## Project Overview
Mechanistic interpretability research on small transformers trained on a **list-copying task**: given `[d1, d2, SEP, MASK, MASK]`, the model must output `[d1, d2, SEP, d1, d2]`. The goal is to understand how SAE features (especially feature 30) control output order.

## Sequence Format
```
[d1, d2, SEP, o1, o2]   # length = list_len * 2 + 1 = 5
```
- `MASK = n_digits` (100), `SEP = n_digits + 1` (101)
- Output positions are the last `list_len` tokens: `[:, list_len + 1:]`
- "Correct" = model reproduces input exactly at output positions (not sorting)

## Accuracy Convention
**Per-token**, not per-sample — each output position counted independently, matching `src/models/utils.py::accuracy()`. A sample where o1 is right but o2 is wrong contributes 0.5 to accuracy. Val accuracy for `2layer_100dig_64d` ≈ 91.45%.

## Naming Conventions
- **Models**: `{n_layers}layer_{n_digits}dig_{d_model}d` → `models/2layer_100dig_64d.pt`
- **SAEs**: `sae_d{dict_size}_k{top_k}_lr{lr}_seed{seed}_{model_name}.pt` → `results/sae_models/`
- **Default SAE** (used for feature 30 analysis): `sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt`
- **SAE checkpoints** include an `act_mean` key used for centering activations — always load it:
  ```python
  sae_checkpoint = torch.load(SAE_PATH, weights_only=False)
  act_mean = sae_checkpoint["act_mean"].to(DEVICE)
  ```

## Standard Notebook Boilerplate
Every notebook/script starts with this sequence — `load_transformer_model` internally calls `configure_runtime` which populates `src/utils/runtime._RUNTIME`:
```python
DEVICE = setup_notebook(seed=42)
model, model_cfg = load_transformer_model(MODEL_NAME, device=DEVICE)
sae, sae_cfg = load_sae(SAE_NAME, model_cfg['d_model'], device=DEVICE)
train_ds, val_ds = get_dataset(n_digits=N_DIGITS, list_len=LIST_LEN, no_dupes=False)
```
All imports come from `from src.utils.nb_utils import *` and `from src.sae import *`.

## Key Source Files
| File | Purpose |
|------|---------|
| `src/utils/nb_utils.py` | `setup_notebook`, `load_transformer_model`, `load_sae` |
| `src/models/utils.py` | `accuracy()` — reference metric |
| `src/data/datasets.py` | `get_dataset()` — generates all `n_digits^list_len` pairs, 80/20 split |
| `src/sae/steering.py` | `get_xovers_df`, `get_output_swap_bounds`, `swap_outputs`, `feature_steering_experiment` |
| `src/sae/activation_collection.py` | `collect_sae_activations`, `collect_attention_weights` |
| `src/utils/runtime.py` | `_RUNTIME` global — read `list_len`, `device` etc. from here after setup |

## Dataset Notes
- `get_dataset(n_digits=100, list_len=2)` → 10,000 total pairs (100²), split 8000/2000
- Do **not** use `train_split=1.0` for evaluation — inflates accuracy ~4%
- `all_ds = ConcatDataset([train_ds, val_ds])` is common for full-dataset analysis; val is the last `len(val_ds)` rows

## SAE / Interpretability Workflow
1. Collect activations at layer 0 SEP token: `collect_sae_activations(..., layer_idx=0, sep_idx=SEP_TOKEN_INDEX)`
2. Feature steering: scale a feature's activation and observe output changes via `feature_steering_experiment`
3. Crossover analysis pipeline (GPU job via `submit_job.sh`):
   - `get_xovers_df` → finds scale values where o1/o2 argmax crosses over
   - `get_output_swap_bounds` → identifies the scale range that produces a swap
   - `swap_outputs` → verifies the swap at the midpoint scale
   - Results saved as CSVs to `results/xover/`

## Environment
- Python 3.11, managed with `uv` (`uv sync`, `uv add`, etc.)
- Always activate `.venv` before running: `source .venv/bin/activate`
- Run scripts with `python3`, not `python`
- Notebooks are `.py` files with `# %%` cell markers (Jupytext-style), kept in `notebooks/`
- Heavy jobs: `bash submit_job.sh` (SLURM)
