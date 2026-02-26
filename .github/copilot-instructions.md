# Copilot Instructions

## Scope and Task Shape
- This repo studies mechanistic behavior in small attention-only transformers on a list-copy task.
- Canonical sequence format is `[d1, d2, SEP, o1, o2]` where outputs must copy inputs (not sort).
- Token IDs are conventionally `MASK = n_digits` and `SEP = n_digits + 1`; output slice is `[:, list_len + 1:]`.

## Core Architecture and Data Flow
- `src/data/datasets.py::get_dataset()` builds all `n_digits^list_len` combinations and returns `(train_ds, val_ds)` with default `train_split=0.8`.
- `src/models/transformer.py` defines custom attention masks (`build_attention_mask`, `attach_custom_mask`) implementing task-specific routing.
- `src/utils/runtime.py::configure_runtime()` sets global `_RUNTIME` values used across model/util code; many helpers assert these are configured.
- `src/utils/nb_utils.py::load_transformer_model()` configures runtime and returns `(model, model_cfg)`; use this as the default loader in analysis code.
- `src/models/utils.py::accuracy()` is per-token accuracy (each output token contributes independently).

## SAE Conventions
- SAE checkpoints in `results/sae_models/` include `state_dict`, `cfg`, and `act_mean`.
- Always load and pass `act_mean` when collecting/patching activations (see `scripts/run_crossover_analysis.py`).
- For feature steering/crossover work, main entry points are in `src/sae/steering.py`: `get_xovers_df`, `get_output_swap_bounds`, `swap_outputs`.

## Canonical Workflows
- Environment: `uv sync` then `source .venv/bin/activate`.
- Train model: `python3 scripts/train_model.py ...` (supports retries until `--min-acc`; saves to `models/`).
- Train SAE: `python3 scripts/train_sae.py --d_sae ... --top_k ... --n_steps ...`.
- Run crossover pipeline: `python3 scripts/run_crossover_analysis.py` (writes CSVs to `results/xover/`).
- Cluster/GPU workflow is captured in `submit_job.sh` (sync env, activate `.venv`, run analysis scripts).

## Project-Specific Patterns
- Prefer imports from `src.utils.nb_utils` and `src.sae` in notebooks/scripts to stay consistent with existing analysis flow.
- Default analyses use full data via `ConcatDataset([train_ds, val_ds])` when exhaustively scanning input space.
- Do not evaluate with `train_split=1.0`; this mixes train data into evaluation and inflates reported accuracy.
- Existing saved-model naming appears in two styles (`2layer_100dig_64d.pt` and timestamped `L*_H*_D*_V*..._acc*.pt`); do not assume one format only.

## Current Baselines and Files
- Common base model: `models/2layer_100dig_64d.pt`.
- Common SAE for feature-30 analysis: `results/sae_models/sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt`.
- Key reference files: `src/data/datasets.py`, `src/models/transformer.py`, `src/models/utils.py`, `src/utils/nb_utils.py`, `src/sae/steering.py`.

## Reproducibility Requirement
- When running experiments, append a concise entry to `EXPERIMENTS.md` with command, output paths, and headline metrics.