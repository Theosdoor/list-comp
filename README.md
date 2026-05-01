# Mechanistic Interpretability of List-Comparison Transformers

This repository implements and analyzes attention-only transformers trained on a list-comparison task, with sparse autoencoder (SAE) feature analysis and mechanistic interpretation.

**Paper:** Farrell, Theo, Patrick Leask, and Noura Al Moubayed. "Order by Scale: Relative‑Magnitude Relational Composition in Attention‑Only Transformers." *Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025.* https://openreview.net/forum?id=vWRVzNtk7W

## Overview

The model learns to compress list representations into a SEP token and then decompose them. The task structure `[d1, d2, SEP, o1, o2]` enables clean mechanistic analysis of information flow through attention layers.

A custom attention mask enforces the causal structure: input tokens write to SEP, and output tokens read only from SEP and their causally prior output positions.

This repository includes:
- **Core training:** Model and SAE training scripts
- **Mechanistic analysis:** Feature steering, crossover analysis, and attention pattern interpretation
- **Figure generation:** Scripts to reproduce all paper results
- **Exploration history:** Archived notebooks and experimental sweeps for reference

## Repository Structure

```
list-comp-priv/
├── scripts/                       # Core paper reproducibility scripts
│   ├── train_model.py             # Train baseline transformer
│   ├── train_sae.py               # Train sparse autoencoder on SEP activations
│   └── run_crossover_analysis.py  # Feature steering & crossover analysis
├── visualization/                 # Figure & table generation scripts
│   ├── make_2layer_table.py       # Reproduce architecture sweep table
│   ├── nb_compare_sae.py          # Compare SAE checkpoints
│   ├── plot_sae_sweep.py          # Plot SAE sweep results
│   ├── nb_model_interp.py         # Attention flow analysis
│   ├── nb_sae_feat_analysis.py    # SAE feature analysis & heatmaps
│   └── special_latents_across_saes.py  # Special feature correlation analysis
├── src/
│   ├── models/
│   │   ├── transformer.py         # Model construction, masking, make_model()
│   │   ├── utils.py               # save/load model, accuracy helpers
│   │   └── train.py               # Training loop
│   ├── data/
│   │   └── datasets.py            # Dataset generation for the list-comparison task
│   ├── sae/
│   │   ├── activation_collection.py  # Hook-based activation extraction
│   │   ├── hooks.py               # TransformerLens hook utilities
│   │   ├── loading.py             # Load SAE checkpoints
│   │   ├── metrics.py             # SAE evaluation metrics (L0, MSE, etc.)
│   │   ├── steering.py            # Feature steering experiments
│   │   ├── reporting.py           # Failure-reason classification & reports
│   │   └── visualization.py       # Activation and feature visualization
│   ├── interpretability/
│   │   └── interp_utils.py        # Attention pattern and residual-stream analysis
│   └── utils/
│       ├── runtime.py             # Global runtime config (list_len, device, etc.)
│       └── nb_utils.py            # Notebook/display helpers
├── models/                        # Saved model checkpoints (.pt)
├── results/                       # SAE checkpoints and analysis results
├── archive/                       # Exploration artifacts (see archive/README.md)
│   ├── notebooks/                 # Rough exploration notebooks
│   ├── exploratory_scripts/       # Superseded/experimental scripts
│   └── sweeps/                    # WandB sweep configuration files
├── EXPERIMENTS.md                 # Log of experimental runs
├── pyproject.toml
└── README.md
```

### Quick Navigation

- **To reproduce paper figures:** See `visualization/README.md`
- **To train models/SAEs:** See `scripts/`
- **To understand exploration history:** See `archive/README.md`


## Installation

Clone the repository and set up the environment using `uv`:

```bash
uv sync
source .venv/bin/activate
```

Copy `.env.example` to `.env` and fill in your credentials (WandB API key, etc.):

```bash
cp .env.example .env
```

## Usage

### Train a Transformer Model

```bash
python3 scripts/train_model.py \
  --n-layers 2 --n-heads 1 --d-model 64 --n-digits 100 \
  --lr 1e-3 --max-steps 100000 --min-acc 0.9 \
  --wandb   # optional: log to Weights & Biases
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--n-layers` | 2 | Number of transformer layers |
| `--d-model` | 64 | Model dimension |
| `--n-digits` | 100 | Vocabulary size (digits) |
| `--list-len` | 2 | Input list length |
| `--wv` / `--wo` | off | Learn W_V / W_O (off = freeze to identity) |
| `--mlp` | off | Include MLP layers (off = attention-only) |
| `--ln` | off | Use layer normalisation |
| `--min-acc` | 0.9 | Minimum val accuracy; retries up to `--max-retries` |

Saved models are written to `models/` with names like `L2_H1_D64_V100_<timestamp>_acc<val_acc>.pt`.

### Train a Sparse Autoencoder (SAE)

```bash
python3 scripts/train_sae.py \
  --d_sae 150 --top_k 4 --n_steps 50000
```

The SAE is trained on SEP-token activations extracted from a pre-trained model. Checkpoints are saved to `sae_checkpoints/`.

### Interpretability & Analysis

```bash
# Full mechanistic interpretability analysis
python3 scripts/nb_model_interp.py

# Feature steering crossover analysis
python3 scripts/run_crossover_analysis.py

# Compare SAE reconstructions (specify folders instead of scanning all)
python3 visualisation/compare_sae.py \
  --sae_folders sweep_k2bsjr0n sae_checkpoints/sweep_tbxyl1y7 sweep_xliz4f19
```

**SAE Comparison Flags:**

| Flag | Description |
|---|---|
| `--sae_folders` | One or more folders to search for SAE checkpoints (space-separated). If not provided, defaults to `sae_checkpoints/` |
| `--model_path` | Override base model for all SAEs |

### Download Checkpoints from WandB

Download artifacts (models or SAE checkpoints) from WandB projects. The following projects are public:
- **SAE Sweep:** https://wandb.ai/theo-farrell99-durham-university/orderbyscale_sae_sweep/
- **Transformer Models:** https://wandb.ai/theo-farrell99-durham-university/order-by-scale/

```bash
# Download all SAEs from the sweep project
python3 scripts/download_wandb_checkpoints.py \
  --entity theo-farrell99-durham-university \
  --project orderbyscale_sae_sweep \
  --artifact_type sae_model \
  --output_dir sae_checkpoints/wandb/

# Download transformer models
python3 scripts/download_wandb_checkpoints.py \
  --entity theo-farrell99-durham-university \
  --project order-by-scale \
  --artifact_type model \
  --output_dir models/wandb/

# Download from specific runs only
python3 scripts/download_wandb_checkpoints.py \
  --entity theo-farrell99-durham-university \
  --project orderbyscale_sae_sweep \
  --runs run1_id run2_id run3_id \
  --output_dir sae_checkpoints/wandb/

# Filter by artifact name
python3 scripts/download_wandb_checkpoints.py \
  --entity theo-farrell99-durham-university \
  --project orderbyscale_sae_sweep \
  --name_filter final \
  --output_dir sae_checkpoints/wandb/
```

Make sure you're logged into WandB: `wandb login`

### WandB Hyperparameter Sweeps

```bash
wandb sweep sweep_configs/<config>.yaml
wandb agent <sweep_id>
```

## Model Architecture

- **Attention-only transformer** (no MLPs by default)
- **2–3 layers** with a single attention head per layer
- **Constrained weights**: W_V and W_O frozen to identity by default (`--wv`/`--wo` to learn them)
- **No biases** by default, no layer normalisation by default
- **Custom attention mask** enforcing the task-specific causal structure:

```
         d1    d2    SEP   o1    o2   ← keys
d1     [ ·    -∞    -∞    -∞    -∞  ]
d2     [ 0    -∞    -∞    -∞    -∞  ]  (layer 0 mask)
SEP    [ 0     0    -∞    -∞    -∞  ]
o1     [-∞    -∞     0    -∞    -∞  ]
o2     [-∞    -∞     0     0    -∞  ]
```

