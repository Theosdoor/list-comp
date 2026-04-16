# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Key - don't forget to use the .venv for python execution. Also ensure all subagents also use rtk

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
- "Special" features are identified via `identify_special_features` in `src/sae/activation_collection.py`: features whose activation correlates strongly (|r| > threshold) with the SEP attention difference `alpha_d1 − alpha_d2`. Requires `collect_attention_patterns` to obtain `alpha_d1_all`/`alpha_d2_all` first.

## Canonical Workflows
- Environment: `uv sync` then `source .venv/bin/activate`.
- Train model: `python3 scripts/nb_train_model.py ...` (supports retries until `--min-acc`; saves to `models/`).
- Train SAE: `python3 scripts/nb_train_sae.py --d_sae ... --top_k ... --n_steps ...`.
- Run crossover pipeline: `python3 scripts/run_crossover_analysis.py [--feature auto] [--threshold 0.5] [--max-features 2] [--report]`
  - Auto mode (default): detects special features via attention-correlation, runs pipeline for up to `--max-features` features.
  - Override mode: `--feature 30` skips detection and runs only that index.
  - Results layout: `results/xover/<sae_name>/special_features.md` (auto mode) and `results/xover/<sae_name>/<feat_idx>/` per feature.
- SAE sweep comparison: `python3 scripts/nb_compare_sae.py` (evaluates all checkpoints in `results/sae_models/`, writes a markdown comparison table).
- WandB sweeps: `wandb sweep sweeps/<config>.yaml` then `wandb agent <sweep_id>` (or `sbatch slurm/submit_2layer_sweep.sh <sweep_id>`).
- Cluster/GPU workflow is captured in `slurm/submit_job.sh` (sync env, activate `.venv`, run analysis scripts).

## Project-Specific Patterns
- Prefer imports from `src.utils.nb_utils` and `src.sae` in notebooks/scripts to stay consistent with existing analysis flow.
- Default analyses use full data via `ConcatDataset([train_ds, val_ds])` when exhaustively scanning input space.
- Do not evaluate with `train_split=1.0`; this mixes train data into evaluation and inflates reported accuracy.
- Existing saved-model naming appears in two styles (`2layer_100dig_64d.pt` and timestamped `L*_H*_D*_V*..._acc*.pt`); do not assume one format only.

## Current Baselines and Files
- Common base model: `models/2layer_100dig_64d.pt`.
- Common SAE: `results/sae_models/sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt` (feature-30 was the manually-identified special feature; auto-detection now finds this automatically).
- Key reference files: `src/data/datasets.py`, `src/models/transformer.py`, `src/models/utils.py`, `src/utils/nb_utils.py`, `src/sae/steering.py`, `src/sae/reporting.py` (failure-reason classification and markdown report generation).

## Reproducibility Requirement
- When running experiments, append a concise entry to `EXPERIMENTS.md` with command, output paths, and headline metrics.

## Attention Mask Architecture
Two masks are built in `build_attention_mask()` and applied via hooks in `attach_custom_mask()`:
- **`mask_bias_l0`** (layer 0): output tokens (`o1`, `o2`) can only self-attend; SEP reads input digits; outputs are further zeroed via `_zero_o_rows` pattern hook.
- **`mask_bias`** (layers 1+): output tokens read from SEP and causally prior outputs; input tokens are blocked from reading outputs.

This enforces the SEP compression bottleneck: information must flow `inputs → SEP (layer 0) → outputs (layers 1+)`.

## SAE Loading Details
- SAE class: `dictionary_learning.trainers.batch_top_k.BatchTopKSAE(activation_dim, dict_size, k)`.
- Use `load_sae(sae_name, d_model)` from `src/utils/nb_utils.py` — handles both legacy (`W_enc`/`b_enc`/`W_dec`/`b_dec`) and new state dict formats automatically.
- Checkpoints contain `state_dict`, `cfg`, and `act_mean`; always retrieve and pass `act_mean` when patching activations.

## Model Config Inference
`src/models/utils.py::infer_model_config(path)` can auto-infer `d_model`, `n_layers`, `n_heads`, `list_len`, `use_wv`, `use_wo`, etc. directly from a checkpoint — useful when loading models whose names don't encode all parameters.

## Development Environment
- Run tests: `.venv/bin/pytest tests/` (pytest is available but coverage is minimal — only `tests/test_make_2layer_table.py` exists).
- To verify changes to the analysis pipeline, run the crossover pipeline on the baseline model/SAE.
- `src/interpretability/interp_utils.py` contains attention-edge ablation and residual-stream analysis helpers used by `scripts/nb_interpret_model.py` and `scripts/nb_model_interp.py`.
- `itda` is a private git dependency (`git+https://github.com/Theosdoor/itda.git`); update with `uv lock --upgrade-package itda`.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->