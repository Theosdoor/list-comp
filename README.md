# Further Experiments for Order by Scale

Further experiments based on the results from the paper:

> Farrell, Theo, Patrick Leask, and Noura Al Moubayed. "Order by Scale: Relative‑Magnitude Relational Composition in Attention‑Only Transformers." Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025.  
> Link: https://openreview.net/forum?id=vWRVzNtk7W

## Overview

This repository implements and analyzes attention-only transformers trained on a list-comparison task. The model learns to compress list representations into a SEP token and then decompose them. The task structure `[d1, d2, SEP, o1, o2]` enables clean mechanistic analysis of information flow through attention layers.

A custom attention mask enforces the causal structure: input tokens write to SEP, and output tokens read only from SEP and their causally prior output positions.

## Repository Structure

```
list-comp-priv/
├── scripts/                   # Runnable entry-points
│   ├── train_model.py         # Train a transformer model
│   ├── train_sae.py           # Train a sparse autoencoder (SAE) on SEP activations
│   ├── interpret_model.py     # Mechanistic interpretability analysis
│   ├── model_interp.py        # Alternative interpretability script
│   ├── run_crossover_analysis.py  # Feature steering crossover analysis
│   ├── sweep_sae.py           # WandB sweep for SAE hyperparameters
│   ├── compare_sae.py         # Compare multiple SAE runs
│   ├── analyze_failure_reasons.py # Diagnose model failure modes
│   └── generate_report_tables.py  # Generate LaTeX/markdown result tables
├── src/
│   ├── models/
│   │   ├── transformer.py     # Model construction, masking, make_model()
│   │   └── utils.py           # save/load model, accuracy helpers
│   ├── data/
│   │   └── datasets.py        # Dataset generation for the list-comparison task
│   ├── sae/
│   │   ├── activation_collection.py  # Hook-based activation extraction
│   │   ├── hooks.py           # TransformerLens hook utilities
│   │   ├── loading.py         # Load SAE checkpoints
│   │   ├── metrics.py         # SAE evaluation metrics (L0, MSE, etc.)
│   │   ├── steering.py        # Feature steering experiments
│   │   └── visualization.py   # Activation and feature visualisation
│   ├── interpretability/
│   │   └── interp_utils.py    # Attention pattern and residual-stream analysis
│   └── utils/
│       ├── runtime.py         # Global runtime config (list_len, device, etc.)
│       └── nb_utils.py        # Notebook/display helpers
├── models/                    # Saved model checkpoints (.pt)
├── results/                   # SAE checkpoints and analysis results
├── notebooks/                 # Exploratory Jupyter notebooks
├── sweep_configs/             # WandB sweep YAML configs
├── data/                      # Generated datasets (auto-created)
├── EXPERIMENTS.md             # Log of individual experiment runs
├── pyproject.toml
└── submit_job.sh              # HPC job submission script
```

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

The SAE is trained on SEP-token activations extracted from a pre-trained model. Checkpoints are saved to `results/sae_models/`.

### Interpretability & Analysis

```bash
# Full mechanistic interpretability analysis
python3 scripts/interpret_model.py

# Feature steering crossover analysis
python3 scripts/run_crossover_analysis.py

# Compare SAE reconstructions
python3 scripts/compare_sae.py
```

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

