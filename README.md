# list-comp

Mechanistic interpretability experiments for small attention-only transformers on a list-copy task.

The canonical task sequence is `[d1, d2, SEP, o1, o2]`. The model sees masked output tokens and must copy the input digits in order. The project studies the SEP-token bottleneck, SAE features on SEP activations, and feature steering/crossover behavior.

## Setup

Install dependencies with `uv` and use the checked-in virtualenv for local commands:

```bash
uv sync
source .venv/bin/activate
```

Copy the environment template if you need WandB or other credentials:

```bash
cp .env.example .env
```

For automated agent work in this repo, shell commands should be prefixed with `rtk`, and Python commands should use `.venv/bin/python` or `.venv/bin/pytest`.

## Repository Layout

- `src/data/` builds list-copy datasets.
- `src/models/` defines the attention-only transformer, custom attention masks, model loading, and training utilities.
- `src/sae/` contains SAE loading, activation collection, metrics, steering, and report generation.
- `src/utils/` contains runtime configuration and notebook/script loading helpers.
- `scripts/` contains runnable analysis and SAE training scripts.
- `slurm/` contains cluster job wrappers.
- `models/` and `sae_checkpoints/` hold local checkpoints.
- `results/` holds generated analysis outputs.
- `paper/` contains the paper source and figures.

## Core Concepts

`src/data/datasets.py::get_dataset()` returns `(train_ds, val_ds)` using an 80/20 split by default. Token IDs conventionally use `MASK = n_digits` and `SEP = n_digits + 1`; the output slice is `[:, list_len + 1:]`.

The custom attention mask in `src/models/transformer.py` enforces a SEP compression bottleneck:

- Layer 0 lets SEP read input digits while output rows are effectively blocked.
- Later layers let output tokens read SEP and causally prior outputs.

This makes information flow through `inputs -> SEP -> outputs`.

## Common Workflows

Train an SAE on the baseline model:

```bash
.venv/bin/python scripts/train_sae.py --sae_type btk --d_sae 150 --top_k 4 --n_steps 50000
```

Run crossover analysis:

```bash
.venv/bin/python scripts/run_crossover_analysis.py --feature auto --threshold 0.5 --max-features 2 --report
```

Compare SAE checkpoints:

```bash
.venv/bin/python scripts/compare_sae.py --best
```

Plot an SAE sweep comparison report:

```bash
.venv/bin/python scripts/plot_sae_sweep.py --report results/compare_sae/sae_comparison_<timestamp>.md
```

Run tests:

```bash
.venv/bin/pytest tests/
```

## Baseline Artifacts

- Common base model: `models/2layer_100dig_64d.pt`
- Common SAE: `sae_checkpoints/sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt`
- Crossover outputs: `results/xover/<sae_name>/`
- SAE comparison outputs: `results/compare_sae/`

## Development Notes

Use `src.utils.nb_utils.load_transformer_model()` for standard model loading in notebooks and analysis scripts; it configures runtime globals before returning `(model, model_cfg)`.

When loading SAE checkpoints, preserve and pass `act_mean` into activation collection, patching, and steering functions. SAE loading is centralized through `src/sae/loading.py` and supports `btk`, `jumprelu`, and `matryoshka` checkpoints.

For evaluation, keep dataset usage explicit:

- Transformer model accuracy should use the held-out validation split from `get_dataset()`.
- SAE evaluation and exhaustive activation scans should use `ConcatDataset([train_ds, val_ds])`.
- Do not use `train_split=1.0` for model evaluation.

When running experiments, record the command, output paths, and headline metrics in the experiment log. Historical entries currently live in `archive/EXPERIMENTS.md`.
