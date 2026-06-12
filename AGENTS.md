# AGENTS.md

This file provides guidance to LLM coding agents when working with code in this repository.

Key:
- Use the project virtualenv for Python execution (`.venv/bin/python`, `.venv/bin/pytest`, or activate `.venv` first).
- Prefix shell commands with `rtk` per the local RTK instructions.
- Ensure any subagents also use `rtk` and the project virtualenv.

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
- SAE checkpoints in `sae_checkpoints/` include `state_dict`, `cfg`, and `act_mean`.
- Always load and pass `act_mean` when collecting/patching activations (see `scripts/run_crossover_analysis.py`).
- For feature steering/crossover work, main entry points are in `src/sae/steering.py`: `get_xovers_df`, `get_output_swap_bounds`, `swap_outputs`.
- "Special" features are identified via `identify_special_features` in `src/sae/activation_collection.py`: features whose activation correlates strongly (|r| > threshold) with the SEP attention difference `alpha_d1 - alpha_d2`. Requires `collect_attention_patterns` to obtain `alpha_d1_all`/`alpha_d2_all` first.
- **Checkpoint selection:** Use `select_checkpoints(paths, use_best=False)` from `src/sae/loading` to filter final/best checkpoint pairs. Default keeps only final; use `use_best=True` to prefer best-val-loss variants when available. Returns `(selected_paths, using_best_set)`.

## Canonical Workflows
- Environment: `uv sync` then `source .venv/bin/activate`.
- Train model: shared training logic lives in `src/models/train.py`; there is no currently tracked `scripts/train_model.py` entry point, so inspect or restore the intended wrapper before documenting/running model-training commands.
- Train SAE: `.venv/bin/python scripts/train_sae.py --sae_type btk --d_sae ... --top_k ... --n_steps ...`.
- Run crossover pipeline: `.venv/bin/python scripts/run_crossover_analysis.py [--feature auto] [--threshold 0.5] [--max-features 2] [--report]`
  - Auto mode (default): detects special features via attention-correlation, runs pipeline for up to `--max-features` features.
  - Override mode: `--feature 30` skips detection and runs only that index.
  - Results layout: `results/xover/<sae_name>/special_features.md` (auto mode) and `results/xover/<sae_name>/<feat_idx>/` per feature.
- SAE sweep comparison: `.venv/bin/python scripts/compare_sae.py [--best]` (evaluates checkpoints in `sae_checkpoints/`, writes reports under `results/compare_sae/`).
- SAE sweep plotting: `.venv/bin/python scripts/plot_sae_sweep.py --report <results/compare_sae/sae_comparison_*.md>`.
- W&B sweep configs and cluster wrappers are not tracked in the clean submission checkout; if local `archive/`, `sweeps/`, or `slurm/` directories are restored, inspect them before documenting/running sweep or cluster commands.

## Project-Specific Patterns
- Prefer imports from `src.utils.nb_utils` and `src.sae` in notebooks/scripts to stay consistent with existing analysis flow.
- Default analyses use full data via `ConcatDataset([train_ds, val_ds])` when exhaustively scanning input space.
- Do not evaluate with `train_split=1.0`; this mixes train data into evaluation and inflates reported accuracy.
- Existing saved-model naming appears in two styles (`2layer_100dig_64d.pt` and timestamped `L*_H*_D*_V*..._acc*.pt`); do not assume one format only.
- Dataset size is capped by `MAX_DATASET_SIZE` in `src/data/datasets.py`; for large `n_digits^list_len`, `get_dataset()` samples instead of fully enumerating unless `max_dataset_size=None`.

## Dataset Evaluation Standard
- Transformer model accuracy should use the held-out validation split from `get_dataset()` with the default `train_split=0.8`.
- SAE evaluation and exhaustive activation scans should use the full input space via `ConcatDataset([train_ds, val_ds])`.
- Do not silently swap these conventions: model accuracy on full train+val is inflated, while SAE metrics on validation-only are incomplete.
- Prefer explicit dataset helper names once they exist in `src/utils/nb_utils.py`; until then, keep the model-vs-SAE dataset convention visible at call sites.

## Current Baselines and Files
- Common base model: `models/2layer_100dig_64d.pt`.
- Common SAE: `sae_checkpoints/sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt`. Additional SAE checkpoints under `sae_checkpoints/` are local/generated artifacts and ignored by git unless explicitly unignored.
- Key reference files: `src/data/datasets.py`, `src/models/transformer.py`, `src/models/utils.py`, `src/utils/nb_utils.py`, `src/sae/loading.py` (SAE loading and checkpoint selection), `src/sae/steering.py`, `src/sae/reporting.py` (failure-reason classification and markdown report generation).

## Reproducibility Requirement
- When running experiments, append a concise entry to the experiment log with command, output paths, and headline metrics if an experiment log exists in the checkout. The clean submission currently does not track `archive/EXPERIMENTS.md`; create or use a root `EXPERIMENTS.md` only if the project owner restores that convention.

## Attention Mask Architecture
Two masks are built in `build_attention_mask()` and applied via hooks in `attach_custom_mask()`:
- **`mask_bias_l0`** (layer 0): output tokens (`o1`, `o2`) can only self-attend; SEP reads input digits; outputs are further zeroed via `_zero_o_rows` pattern hook.
- **`mask_bias`** (layers 1+): output tokens read from SEP and causally prior outputs; input tokens are blocked from reading outputs.

This enforces the SEP compression bottleneck: information must flow `inputs -> SEP (layer 0) -> outputs (layers 1+)`.

## SAE Loading Details
- SAE classes are instantiated centrally in `src/sae/loading.py::instantiate_sae_from_cfg()`; supported `sae_type` values are `btk`, `jumprelu`, and `matryoshka`.
- Use `load_sae(sae_path, d_model)` from `src/utils/nb_utils.py`; it delegates to `src/sae/loading.py` and handles legacy (`W_enc`/`b_enc`/`W_dec`/`b_dec`) and new state dict formats automatically.
- Checkpoints contain `state_dict`, `cfg`, and `act_mean`; always retrieve and pass `act_mean` when patching activations.

## Model Config Inference
`src/models/utils.py::infer_model_config(path)` can auto-infer `d_model`, `n_layers`, `n_heads`, `list_len`, `use_wv`, `use_wo`, etc. directly from a checkpoint; this is useful when loading models whose names don't encode all parameters.

## Development Environment
- Run tests: `.venv/bin/pytest tests/`. Current tests cover datasets, SAE hooks/metrics/reporting/loading, and SAE sweep plotting.
- To verify changes to the analysis pipeline, run the crossover pipeline on the baseline model plus the included baseline SAE checkpoint, or on another local/generated SAE checkpoint when relevant.
- `src/interpretability/interp_utils.py` contains attention-edge ablation and residual-stream analysis helpers used by exploratory notebooks/scripts.
- Private dependency sources should stay omitted from blind-review materials.
