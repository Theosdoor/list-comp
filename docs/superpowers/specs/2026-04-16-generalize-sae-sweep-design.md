# Generalise SAE Sweep Script — Design Spec

**Date:** 2026-04-16  
**Status:** Approved

## Overview

Extend `scripts/sweep_sae.py` and the SAE loading utilities to support multiple SAE architectures (BatchTopK, JumpReLU, Matryoshka) through a shared script and a registry pattern, while keeping each architecture's sweep parameters in its own W&B sweep YAML.

## Motivation

The current codebase hardcodes `BatchTopKSAE` / `BatchTopKTrainer` throughout the training and loading pipeline. To compare JumpReLU and Matryoshka SAEs against BTK on the list-copy task (and to test whether the crossover pipeline generalises), the infrastructure needs to dispatch on SAE type without duplicating the shared training loop, metric collection, or checkpoint format.

---

## Architecture

### Trainer Registry (`scripts/sweep_sae.py`)

A module-level registry maps type strings to factory functions:

```python
TRAINER_REGISTRY = {
    "btk":        _make_btk_trainer,
    "jumprelu":   _make_jumprelu_trainer,
    "matryoshka": _make_matryoshka_trainer,
}
```

Adding a new type (e.g. from `itda`) requires only: one factory function + one registry entry.

Each factory has the signature:

```python
def _make_<type>_trainer(config, activation_dim: int, device: str) -> tuple[SAETrainer, str, dict]:
    ...
    return trainer, run_name, extra_cfg
```

- `config` — `wandb.config` object
- `run_name` — human-readable string prefixed with SAE type (e.g. `btk_sae_d256_k3_seed0`)
- `extra_cfg` — type-specific fields written into the checkpoint `cfg` block

`train_sae_sweep()` reads `sae_type = wandb.config.sae_type`, looks it up in the registry, and calls the factory. Everything else — SEP activation collection, the training step loop, metric logging, checkpoint saving, W&B artifact upload — is shared and unchanged.

### Save Folder

The save folder is derived dynamically from the W&B sweep ID after `wandb.init()`:

```python
sweep_id = run.sweep_id or "standalone"
SAVE_FOLDER = f"results/sae_models/sweep_{sweep_id}"
```

All runs within a sweep land in the same folder. Standalone runs fall back to `sweep_standalone`.

### Naming Convention

`sae_type` is prepended to all identifiers:

| Identifier | Example |
|---|---|
| Checkpoint filename | `btk_sae_d256_k3_lr1e-4_seed0_2layer_100dig_64d.pt` |
| W&B run name | `btk_sae_d256_k3_seed0` |
| W&B artifact | `btk-sae-d256-k3-seed0` |

Type-specific filename tokens:
- BTK: `_k{top_k}_`
- JumpReLU: `_tl0{target_l0}_`
- Matryoshka: `_k{top_k}_ng{n_groups}_`

Existing BTK checkpoints (no `sae_type` in `cfg`) continue to load correctly via backward-compat default.

---

## Checkpoint Format

The `cfg` block in every checkpoint gains a `sae_type` field. Existing checkpoints without it default to `"btk"`.

Type-specific fields stored in `cfg`:

**BTK:** `k`, `dict_size`, `activation_dim`, `lr`, `seed`  
**JumpReLU:** `target_l0`, `sparsity_penalty`, `bandwidth`, `dict_size`, `activation_dim`, `lr`, `seed`  
**Matryoshka:** `k`, `n_groups`, `group_sizes`, `group_fractions`, `dict_size`, `activation_dim`, `lr`, `seed`

---

## Loading

### Single dispatch point

`src/sae/loading.py` gains:

```python
def instantiate_sae_from_cfg(cfg: dict, d_model: int, device: str) -> nn.Module:
    sae_type = cfg.get("sae_type", "btk")  # backward compat default
    ...
```

This is the only place in the codebase that switches on `sae_type` for instantiation.

### Affected loaders

All three existing loaders call `instantiate_sae_from_cfg` instead of constructing SAEs inline:

| File | Function | Change |
|---|---|---|
| `src/sae/loading.py` | `load_sae_from_local` | delegate instantiation |
| `src/sae/loading.py` | `load_sae_from_wandb_run` | delegate instantiation; update artifact name pattern |
| `src/utils/nb_utils.py` | `load_sae` | delegate instantiation |
| `scripts/nb_compare_sae.py` | local `load_sae` | remove local copy; import from `src.sae.loading` |

### Crossover pipeline compatibility

`run_crossover_analysis.py` calls `load_sae` from `nb_utils`, which will dispatch correctly once the loader is updated. The downstream functions (`collect_sae_activations`, `identify_special_features`, `get_xovers_df`) call `sae.encode(x)` and `sae(x)` — all three SAE classes implement the `Dictionary` interface, so no changes needed there. Compatibility should be verified empirically by running the crossover pipeline on a JumpReLU checkpoint after training.

---

## Sweep YAMLs

All three YAMLs point to `program: scripts/sweep_sae.py`.

### `sweeps/btksae_sweep.yaml` (updated)
Add `sae_type: value: btk`. All other parameters unchanged.

### `sweeps/jumprelu_sweep.yaml` (new)
```yaml
parameters:
  sae_type:    { value: jumprelu }
  d_sae:       { values: [128, 192, 256, 320] }
  target_l0:   { values: [2, 3, 4, 5, 6] }
  sparsity_penalty: { value: 1.0 }
  lr:          { value: 7e-5 }
  n_steps:     { value: 75000 }
  warmup_steps: { value: 1000 }
  batch_size:  { value: 4096 }
  seed:        { values: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] }
```

`target_l0` covers the same effective range as BTK's `top_k: [2..6]` for clean cross-type comparison. `lr: 7e-5` follows the DeepMind JumpReLU paper default.

### `sweeps/matryoshka_sweep.yaml` (new)
```yaml
parameters:
  sae_type:    { value: matryoshka }
  d_sae:       { values: [128, 192, 256, 320] }
  top_k:       { values: [3, 4, 5] }
  n_groups:    { values: [2, 4] }
  lr:          { value: 1e-4 }
  n_steps:     { value: 75000 }
  warmup_steps: { value: 1000 }
  batch_size:  { value: 4096 }
  seed:        { values: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] }
```

`n_groups` is swept over `[2, 4]`; equal `group_fractions = [1/n_groups] * n_groups` are derived in the factory function.

---

## Files Changed

| File | Type of change |
|---|---|
| `scripts/sweep_sae.py` | Add registry + 3 factory functions; dynamic save folder |
| `sweeps/btksae_sweep.yaml` | Add `sae_type: value: btk` |
| `sweeps/jumprelu_sweep.yaml` | New file |
| `sweeps/matryoshka_sweep.yaml` | New file |
| `src/sae/loading.py` | Add `instantiate_sae_from_cfg`; update both loaders |
| `src/utils/nb_utils.py` | Update `load_sae` to delegate instantiation |
| `scripts/nb_compare_sae.py` | Remove local `load_sae`; import from `src.sae.loading` |

---

## Out of Scope

- `itda`-based SAE types (registry is designed to accept them; implementation deferred)
- Changes to crossover analysis logic, steering, or reporting
- Retraining existing BTK checkpoints in new naming format
