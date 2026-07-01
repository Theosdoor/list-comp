# Sparse Autoencoders Can Learn Graded Latents for Relational Composition

Code for the paper:
> Farrell, Theo, Patrick Leask, and Noura Al Moubayed. "Sparse Autoencoders Can Learn Graded Latents for Relational Composition." Mechanistic Interpretability Workshop at ICML 2026. https://openreview.net/forum?id=ltQ6XAduTF

<div align="center">
  <img src="figures/figure1_heatmap.svg" alt="Figure 1: Heatmap of latent activations" width="80%"/>
</div>

<div align="center">

> Figure 1: These $100 \times 100$ heatmaps show example SAE latent activations over all input bigrams $(d_1, d_2)$ with $d_1, d_2 \in [0, 99]$: top-left cell is $(0,0)$, bottom-right cell is $(99,99)$. **Left** (1-symbol detector): latent 0 activates strongly if $d_1$ or $d_2 = 41$ and peaks at $d_1 = d_2 = 41$, producing a plus-sign pattern. **Centre** ($k$-symbol detector, $k > 1$): latent 5 activates strongly if $d_1$ or $d_2 \in \{7, 58\}$, producing two plus-sign patterns. **Right** (special latent): latent 11 activates on most inputs with generally much greater magnitudes than any other latent.

</div>

This repository contains the code and core artifacts for mechanistic interpretability experiments on a small attention-only transformer trained on a list-copy task. The experiments study whether sparse autoencoders (SAEs) trained on the SEP-token residual stream learn graded features that mediate relational composition.

The transformer code and baseline transformer model build on the companion repository [Order-by-Scale](https://github.com/Theosdoor/Order-by-Scale). This repository adds the SAE training/loading code, special-feature detection, feature steering, crossover analysis, reporting, plotting, and submission-specific experiment scripts.

The canonical sequence is:

```text
[d1, d2, SEP, o1, o2]
```

The model receives masked output tokens and must copy the input digits in order. The custom attention mask creates a SEP-token bottleneck, so information flows through:

```text
input digits -> SEP token -> output tokens
```

## Reviewer Quick Start

For a clean checkout, the fastest way to verify the codebase is:

```bash
uv sync
source .venv/bin/activate
pytest tests/
python -c "import src; import src.utils.nb_utils; print('ok')"
```

The tracked repository includes the baseline transformer checkpoint at `models/2layer_100dig_64d.pt` and the baseline BatchTopK SAE checkpoint at `sae_checkpoints/sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt`. Additional SAE checkpoints and generated result tables are treated as local or regenerated artifacts.

## Setup

The project uses `uv` for dependency management.

```bash
uv sync
source .venv/bin/activate
```

Run commands from the repository root with the virtual environment active.

If you want to use Weights & Biases for SAE training or checkpoint downloads, create a local environment file:

```bash
cp .env.example .env
```

Then add your own W&B values to `.env`. The submitted code and local scripts do not require W&B unless you pass W&B-specific options.

## Included Artifacts

The cleaned submission includes the code, tests, and baseline transformer needed to inspect and rerun the pipeline:

- `models/2layer_100dig_64d.pt`: baseline attention-only transformer checkpoint.
- `sae_checkpoints/sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt`: baseline BatchTopK SAE checkpoint.
- `src/`: source code for datasets, models, SAE loading, metrics, steering, reporting, and utilities.
- `scripts/`: runnable training and analysis entry points.
- `tests/`: regression tests for datasets, SAE utilities, loading, reporting, and plotting.
- `visualisation/nb_sae_feat_analysis.ipynb`: notebook used for SAE feature exploration.

The following directories are intentionally local or regenerated and may not exist in a clean clone:

- Additional files under `sae_checkpoints/`: SAE checkpoints produced by `scripts/train_sae.py` or downloaded from W&B.
- `results/`: generated comparison reports, crossover CSVs, markdown failure reports, and plots.
- `wandb/`: local W&B run metadata.

## Repository Layout

- `src/data/`: exhaustive list-copy dataset construction.
- `src/models/`: transformer definition, custom attention masks, model loading, and training utilities.
- `src/sae/`: SAE checkpoint loading, activation collection, metrics, steering, crossover analysis, and markdown reporting.
- `src/interpretability/`: attention-edge ablation and residual-stream analysis helpers.
- `src/utils/`: runtime configuration and notebook/script loading helpers.
- `scripts/`: CLI scripts for SAE training, evaluation, plotting, and crossover analysis.

## Task and Evaluation Conventions

`src/data/datasets.py::get_dataset()` builds digit combinations and returns `(train_ds, val_ds)` with an 80/20 split by default. Token IDs conventionally use:

- `MASK = n_digits`
- `SEP = n_digits + 1`
- output slice: `[:, list_len + 1:]`

Transformer model accuracy should be evaluated on the held-out validation split. SAE metrics and exhaustive activation scans generally use the full input space via:

```python
ConcatDataset([train_ds, val_ds])
```

Do not use `train_split=1.0` for model evaluation, because that mixes train data into the reported accuracy.

## Quick Verification

Run the test suite:

```bash
pytest tests/
```

Check that the main imports work from the repository root:

```bash
python -c "import src; import src.utils.nb_utils; print('ok')"
```

To check that the tracked baseline transformer can be loaded and its config inferred:

```bash
python -c "from src.models.utils import infer_model_config; print(infer_model_config('models/2layer_100dig_64d.pt', device='cpu'))"
```

## Script Guide

All examples assume they are run from the repository root.

### Reproduce the Analysis Flow

The main end-to-end path is:

1. Verify the baseline transformer and tests.
2. Inspect or reuse the included baseline SAE checkpoint, or train/download additional checkpoints into `sae_checkpoints/`.
3. Inspect checkpoint metadata with `scripts/inspect_saes.py`.
4. Compare SAE checkpoints with `scripts/compare_sae.py`.
5. Run crossover analysis for automatically detected special features with `scripts/run_crossover_analysis.py`.
6. Plot sweep summaries from a generated comparison report with `scripts/plot_sae_sweep.py`.

### Inspect Checkpoints

Print checkpoint metadata, config fields, stored metrics, `act_mean`, and state-dict tensor shapes:

```bash
python scripts/inspect_saes.py sae_checkpoints/sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt
```

You can pass a directory to inspect all `.pt` files below it:

```bash
python scripts/inspect_saes.py sae_checkpoints/
```

### Train an SAE

Train a new SAE on SEP-token activations from the baseline transformer:

```bash
python scripts/train_sae.py \
  --model_path models/2layer_100dig_64d.pt \
  --sae_type btk \
  --d_sae 128 \
  --top_k 3 \
  --n_steps 50000 \
  --seed 1
```

Supported SAE types are:

- `btk`: BatchTopK SAE, using `--top_k`.
- `jumprelu`: JumpReLU SAE, using `--target_l0` and `--sparsity_penalty`.
- `matryoshka`: Matryoshka BatchTopK SAE, using `--top_k` and `--n_groups`.

Outputs are saved under `sae_checkpoints/` by default. Each checkpoint stores the SAE state dict, config, and `act_mean`; keep `act_mean` with the checkpoint because downstream patching and steering use it.

To run with W&B sweep config injection:

```bash
python scripts/train_sae.py --wandb
```

### Compare SAE Checkpoints

Evaluate one or more SAE checkpoint folders and write a markdown comparison report:

```bash
python scripts/compare_sae.py \
  --sae-folders sae_checkpoints/ \
  --model-path models/2layer_100dig_64d.pt
```

Useful options:

- `--best`: prefer best-validation-loss checkpoints when both final and best variants exist.
- `--special-threshold 0.5`: threshold for identifying attention-correlated special features.
- `--output-dir results/compare_sae/`: destination for reports and tables.
- `--exclude-l0`, `--exclude-d-sae`: filter sweep summaries.

The report includes sparsity, dead-feature rate, reconstruction quality, downstream loss recovery, and special-feature counts.

### Plot SAE Sweep Summaries

Convert a `compare_sae.py` markdown report into aggregate tables and figures:

```bash
python scripts/plot_sae_sweep.py \
  --report results/compare_sae/sae_comparison_<timestamp>.md
```

By default, outputs are written to `results/compare_sae/figures/`. The script can filter by L0 or dictionary size and can omit table error bars or selected columns.

### Run Crossover Analysis

Run the feature steering and output-swap pipeline for automatically detected special features:

```bash
python scripts/run_crossover_analysis.py \
  --model 2layer_100dig_64d \
  --sae sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt \
  --feature auto \
  --threshold 0.5 \
  --max-features 2 \
  --report
```

You can also target a specific SAE feature:

```bash
python scripts/run_crossover_analysis.py \
  --sae sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt \
  --feature 30
```

Outputs are written under:

```text
results/xover/<sae_name>/<feature_idx>/
```

The main CSV outputs are:

- `xovers_feat<idx>.csv`: feature steering crossover points.
- `swap_bounds_feat<idx>.csv`: inferred output-swap regions.
- `swap_results_feat<idx>.csv`: verification of output swaps.
- `failure_analysis_feat<idx>.md`: optional markdown report when `--report` is used.

### Analyse Special Latents Across SAEs

Generate plots and reports relating SAE quality metrics to special-feature counts:

```bash
python scripts/special_latents_across_saes.py \
  --sae_dirs sae_checkpoints/ \
  --model_path models/2layer_100dig_64d.pt \
  --alpha_diff_thresh 0.5 \
  --output_dir results/sae_plots
```

This computes loss recovered, explained variance, actual L0, and attention-correlation statistics across all selected checkpoints.

### Download W&B Sweep Checkpoints

If you trained SAEs with W&B artifacts, download all model artifacts from a sweep:

```bash
python scripts/download_wandb_checkpoints.py <sweep_id> \
  --project <entity/project> \
  --save_dir sae_checkpoints
```

This requires W&B credentials in the environment or in `.env`.

## Programmatic Loading

Use the notebook utilities for standard loading:

```python
from src.utils.nb_utils import load_transformer_model, load_sae

model, model_cfg = load_transformer_model("2layer_100dig_64d")
sae, sae_cfg = load_sae(
    "sae_checkpoints/sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt",
    d_model=model_cfg["d_model"],
)
```

`load_transformer_model()` configures runtime globals before returning `(model, model_cfg)`. `load_sae()` delegates to the centralized SAE loader and supports current and legacy checkpoint formats.

## Notes for Reproduction

- Run commands from the repository root.
- Keep model evaluation on validation data and SAE scans on the full train-plus-validation input space.
- Preserve `act_mean` when moving SAE checkpoints.
- The default scripts choose CUDA when available and fall back to CPU where supported.

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@inproceedings{
farrell2026sparse,
title={Sparse Autoencoders Can Learn Graded Latents for Relational Composition},
author={Theo Farrell and Patrick Leask and Noura Al Moubayed},
booktitle={Mechanistic Interpretability Workshop at ICML 2026},
year={2026},
url={https://openreview.net/forum?id=ltQ6XAduTF}
}
```
