# AI Agent Instructions: List Comprehension Transformer Project

## Project Overview
This project trains small transformer models to learn list comparison tasks (e.g., `[d1, d2] -> [o1, o2]` where outputs copy inputs in sorted order) and analyzes them using Sparse Autoencoders (SAEs). The goal is to understand mechanistic interpretability through controlled experiments.

## Critical Runtime Configuration Pattern
**All model/data operations require runtime configuration first:**
```python
from src.utils.runtime import configure_runtime

configure_runtime(
    list_len=2,           # Number of input digits
    seq_len=5,            # list_len * 2 + 1 (includes SEP token)
    vocab=102,            # n_digits + 2 (mask_tok and sep_tok)
    device="cuda",        # or "mps"/"cpu"
    seed=42
)
```
Without this, functions using `_RUNTIME` (in [src/models/transformer.py](src/models/transformer.py), [src/models/utils.py](src/models/utils.py)) will fail with assertion errors. Always call `configure_runtime()` before `make_model()`, `build_attention_mask()`, or `attach_custom_mask()`.

## Architecture: Sequence Format & Custom Attention
Input sequences follow strict format: `[d1, d2, SEP, o1, o2]` where:
- `d1, d2`: Input digits (0-99 for standard vocab)
- `SEP`: Separator token (position `list_len`, index `n_digits + 1`)
- `o1, o2`: Output positions (masked during training with `mask_tok = n_digits`)

**Custom attention masks enforce task structure** (see [src/models/transformer.py](src/models/transformer.py#L25-L50)):
- Layer 0: Output tokens can only self-attend; SEP can read inputs
- Layer 1+: Full causal attention except outputs cannot attend to inputs
- Implemented via `attach_custom_mask()` using permanent hooks on `hook_attn_scores`

## Model Naming Conventions
Two formats are parsed by `parse_model_name()` in [src/models/transformer.py](src/models/transformer.py#L206-L267):
1. **New format**: `L{layers}_H{heads}_D{d_model}_V{vocab}[_len{list_len}][_flags][_timestamp]`
   - Example: `L3_H1_D64_V100_len3_260121-143443_acc0.9962.pt`
2. **Old format**: `{layers}layer_{digits}dig_{d_model}d[_...]`
   - Example: `2layer_100dig_64d.pt` (always assumes `list_len=2`)

Use `parse_model_name_safe()` for robust parsing with fallback.

## Accuracy Calculation
**Model accuracy must only be evaluated on output positions, NOT all tokens:**

```python
# CORRECT (from src/models/utils.py):
logits = model(inputs)[:, list_len + 1:]  # Only output positions
preds = logits.argmax(-1)
hits = (preds == targets[:, list_len + 1:]).sum().item()
accuracy = hits / preds.numel()

# WRONG - evaluates all tokens including inputs and SEP:
logits = model(inputs)  # ❌ Don't do this
preds = logits.argmax(-1)
hits = (preds == targets).sum().item()  # ❌ Wrong denominator
```

**Why:** Inputs are provided to the model (d1, d2), and SEP is a special token. Only the output positions (o1, o2) should be predicted and evaluated. For `list_len=2`, these are positions 3 and 4 (indices after the SEP at position 2).

**Reference implementation:** [src/models/utils.py](src/models/utils.py#L17-L30) `accuracy()` function.

## Source Code Structure

### `src/data/`
- **`datasets.py`**: Dataset generation and loading
  - `get_dataset()` - Main function for creating train/val splits with duplicate handling
  - `ListDataset` - PyTorch Dataset for list comparison tasks
  - Supports caching to disk, configurable duplicate behavior

### `src/models/`
- **`transformer.py`**: Core transformer architecture
  - `make_model()` - Factory function for creating HookedTransformer models
  - `build_attention_mask()` - Constructs custom causal masks for task structure
  - `attach_custom_mask()` - Applies permanent hooks to enforce attention patterns
  - `parse_model_name()` / `parse_model_name_safe()` - Parse model filenames to extract config
- **`utils.py`**: Model utilities
  - `accuracy()` - Evaluate model on validation set (output positions only!)
  - `save_model()` / `load_model()` - Checkpoint management
  - `infer_model_config()` - Reverse-engineer model architecture from checkpoint

### `src/sae/`
- **`sae_analysis.py`**: SAE training and analysis utilities
  - `collect_sae_activations()` - Extract SAE latent activations for all validation data
  - `create_feature_heatmaps()` - Visualize feature activation patterns across input space
  - `compute_reconstruction_metrics()` - MSE, explained variance for SAE reconstructions
  - `compute_sae_reconstruction_accuracy()` - Model accuracy when using SAE-reconstructed activations
  - `identify_special_features()` - Find features correlated with attention patterns
  - `load_sae_from_local()` / `load_sae_from_wandb_run()` - Load trained SAE checkpoints

### `src/utils/`
- **`runtime.py`**: Global configuration singleton
  - `configure_runtime()` - Set global task parameters (list_len, seq_len, vocab, device)
  - `_RUNTIME` - Global config object accessed by models and utilities
- **`nb_utils.py`**: Notebook convenience functions
  - `setup_notebook()` - Initialize device, seeds, gradient settings
  - `load_transformer_model()` - Load model with automatic config parsing
  - `load_sae()` - Load SAE with d_model parameter
  - Cleaner API than raw loading functions

### `src/interpretability/`
- **`interp_utils.py`**: Model interpretability tools
  - Attention pattern analysis
  - Activation patching utilities
  - Feature visualization helpers

## SAE Training & Analysis
SAEs use BatchTopK architecture from `dictionary_learning` library (see [scripts/train_sae.py](scripts/train_sae.py)):
- Target SEP token activations: `blocks.{layer_idx}.hook_resid_post[:, sep_idx, :]`
- Config follows "Order by Scale" paper: `top_k=4`, `d_sae=100-256`, `lr=3e-4`
- Analysis functions in [src/sae/sae_analysis.py](src/sae/sae_analysis.py): `collect_sae_activations()`, `create_feature_heatmaps()`

**SAE model naming**: `sae_d{d_sae}_k{top_k}_lr{lr}_seed{seed}_{base_model_name}.pt`

## Dataset Behavior
`get_dataset()` in [src/data/datasets.py](src/data/datasets.py) has critical flags:
- `no_dupes=True`: Filter out duplicate inputs (d1==d2) entirely
- `train_dupes_only=True`: Train on all data, validate only on non-duplicates
- Both False (default): Include duplicates in both train/val splits

**Always** match dataset config to how the model was trained when evaluating.

### Critical: Validation Split for Accurate Metrics
**DO NOT use `train_split=1.0` when evaluating models** - this inflates accuracy by ~4% because it includes training data.

For proper validation:
```python
# CORRECT - uses default train_split=0.8 (matches training)
_, val_ds = get_dataset(list_len=2, n_digits=100)

# WRONG - evaluates on ALL data including training set
val_ds, _ = get_dataset(list_len=2, n_digits=100, train_split=1.0)
```

True validation accuracy for `2layer_100dig_64d.pt` is **91.45%**, not ~95% (which is train+val combined).

## Development Workflows

### Training a model
```bash
cd scripts
python train_model.py --n-layers 2 --d-model 64 --n-digits 100 \
    --lr 1e-3 --max-steps 100000 --early-stop-acc 0.999 \
    --wandb  # Optional W&B logging
```
Models save to `models/` with auto-generated names. Use `--name` to override.

### Training SAE sweep
```bash
# Run grid search across hyperparameters
wandb sweep sweep_configs/sweep.yaml  # Get sweep ID
wandb agent <sweep-id>
```
Results save to `results/sae_models/`. Compare with `scripts/compare_sae.py`.

### Notebook setup pattern
Standard imports for analysis notebooks (see [src/utils/nb_utils.py](src/utils/nb_utils.py)):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # Add project root to path

from src.utils.nb_utils import setup_notebook, load_model, load_sae
device = setup_notebook(seed=42, disable_grad=True)

# Then load model/SAE using convenience functions
model = load_model('2layer_100dig_64d', 'models/')
```

## Key Dependencies
- `transformer-lens`: Base transformer implementation (HookedTransformer)
- `dictionary_learning`: SAE training (BatchTopKSAE, BatchTopKTrainer)
- `wandb`: Experiment tracking (enable with `--wandb` flag)
- `uv`: Package manager (use `uv add` not `pip install`)

## Common Pitfalls
1. **Forgetting `configure_runtime()`**: Always call before model operations
2. **Mismatched `list_len`**: Ensure dataset, mask, and model use same value
3. **SEP token indexing**: Position is `list_len` (e.g., index 2 for list_len=2)
4. **Old vs new notebooks**: Check imports—old code may use deprecated paths like `model_scripts.model_utils`
5. **Device consistency**: Match model device with data in `DataLoader` or use `.to(device)`

## File Organization
- `src/`: Reusable modules (models, data, SAE analysis, utilities)
- `scripts/`: Training/sweep scripts runnable from CLI
- `notebooks/`: Analysis notebooks (dated by creation, e.g., `nb_2026_02_04.py`)
- `results/`: Trained SAE models and sweep outputs
- `models/`: Trained transformer checkpoints
- `data/`: Pre-generated datasets (`.pt` files)

## Experiment Logging (Required)
**Always log experiments to `EXPERIMENTS.md` for reproducibility.** After training models or SAEs:

```bash
python scripts/log_experiment.py \
    --title "SAE sweep BatchTopK k=4" \
    --command "wandb sweep sweep_configs/sweep.yaml && wandb agent <sweep-id>" \
    --outputs "results/sae_models/" \
    --results "Best: sae_d100_k4 with MSE=0.0044" \
    --notes "Testing d_sae scaling from 50-256"
```

For manual experiments, either use the script or add entries directly to `EXPERIMENTS.md` with: command, outputs, key results, git commit.