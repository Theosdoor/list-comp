# GEMINI.md

This file provides foundational guidance for Gemini when working in this repository. 

## Project Overview
This repository focuses on the **mechanistic interpretability** of small, attention-only transformers trained on a **list-copy task**. The core research explores how these models compress list representations into a `SEP` token and subsequently decompose them to produce outputs.

### Task & Data Format
- **Canonical Sequence:** `[d1, d2, SEP, o1, o2]`.
- **Goal:** Copy inputs (not sort). `o1` should match `d1`, `o2` should match `d2`.
- **Tokens:** `MASK = n_digits`, `SEP = n_digits + 1`.
- **Output Slice:** `[:, list_len + 1:]`.

## Core Architecture & Data Flow
- **Attention-Only:** No MLPs by default.
- **SEP Bottleneck:** Information must flow `inputs → SEP (Layer 0) → outputs (Layer 1+)`.
- **Custom Attention Masks** (defined in `src/models/transformer.py`):
    - **`mask_bias_l0` (Layer 0):** `SEP` reads input digits; output tokens (`o1`, `o2`) can only self-attend.
    - **`mask_bias` (Layers 1+):** Output tokens read from `SEP` and causally prior outputs; input tokens are blocked from reading outputs.
- **Runtime Configuration:** Managed via `src/utils/runtime.py`. Helpers like `configure_runtime` set global values (`list_len`, `device`, etc.) used across the codebase.

## Sparse Autoencoders (SAEs)
- **Class:** `dictionary_learning.trainers.batch_top_k.BatchTopKSAE`.
- **Training Target:** Trained on `SEP` token activations from Layer 0.
- **"Special" Features:** Identified in `src/sae/activation_collection.py` by correlating latent activations with the attention difference `alpha_d1 - alpha_d2` at the `SEP` token. A threshold (default 0.5) is used.
- **Crossover Analysis:** Scaling special features to swap model outputs (e.g., forcing `(d1, d2) → (d2, d1)`).
    - **`o1` Crossover:** Detected via **linear fit** (highly linear empirically).
    - **`o2` Crossover:** Detected via **grid search + bisection** (nonlinear behavior).

## Canonical Workflows & Commands
**Golden Rule:** Always prefix commands with `rtk` (Rust Token Killer) for token optimization.

### Environment Setup
```bash
rtk uv sync
source .venv/bin/activate
```

### Training
- **Model:** `rtk python3 scripts/train_model.py --n-layers 2 --n-heads 1 --d-model 64 --n-digits 100 --min-acc 0.9`
- **SAE:** `rtk python3 scripts/train_sae.py --d_sae 150 --top_k 4 --n_steps 50000`

### Analysis & Reproduction
- **Crossover Pipeline:** `rtk python3 scripts/run_crossover_analysis.py [--feature auto] [--threshold 0.5] [--max-features 2] [--report]`
- **Model Interpretation:** `rtk python3 scripts/nb_model_interp.py`
- **SAE Comparison:** `rtk python3 visualisation/compare_sae.py`
- **Tests:** `rtk .venv/bin/pytest tests/`

## Development Conventions
- **SAE Loading:** Use `load_sae(sae_name, d_model)` from `src/utils/nb_utils.py`. It handles multiple state-dict formats.
- **Checkpoint Selection:** Use `select_checkpoints(paths, use_best=False)` from `src/sae/loading` to filter final/best checkpoint pairs without duplication.
- **Activation Collection:** Always retrieve and pass `act_mean` when collecting/patching activations.
- **Inference:** Use `src/models/utils.py::infer_model_config(path)` to auto-detect architecture from checkpoints.
- **Reporting:** Crossover results are saved in `results/xover/<sae_name>/`. `src/sae/reporting.py` handles failure-reason classification.
- **Reproducibility:** Append a concise entry to `EXPERIMENTS.md` after running experimental scripts.

