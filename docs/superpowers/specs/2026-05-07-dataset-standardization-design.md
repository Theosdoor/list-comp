# Dataset Standardization Design

**Date:** 2026-05-07  
**Problem:** Inconsistent dataset usage across evaluation contexts (SAE evaluation uses full train+val data, but utility functions default to validation-only, and some scripts mix patterns).  
**Goal:** Create explicit, standardized helpers that make dataset choice clear and prevent mistakes.

## Rationale

- **SAE evaluation** requires the full dataset (train + validation) because SAEs are trained on the complete data and should be evaluated exhaustively
- **Model accuracy evaluation** requires a held-out validation split to properly measure generalization
- Current code has both patterns mixed throughout, making it unclear which to use and error-prone for new analysis scripts

## Solution: Explicit Helper Functions

Add two dedicated utilities in `src/utils/nb_utils.py`:

### `get_eval_dataset_sae()`
```python
def get_eval_dataset_sae(
    list_len=2, 
    n_digits=100, 
    batch_size=2048, 
    no_dupes=False,
    seed=0,
    device=None
):
    """
    Get full (train + validation) dataset and dataloader for SAE evaluation.
    
    SAEs are trained on the complete dataset (both train and validation splits),
    so evaluation must also use the full dataset to properly measure performance.
    
    Returns:
        (full_dataset, dataloader) where full_dataset is ConcatDataset([train_ds, val_ds])
    """
```

**Key properties:**
- Returns `ConcatDataset([train_ds, val_ds])` for exhaustive evaluation
- Default batch_size=2048 (suitable for SAE metrics computation)
- Passes through `no_dupes` and other `get_dataset()` kwargs
- Self-documenting name makes the intent clear

### `get_eval_dataset_model()`
```python
def get_eval_dataset_model(
    list_len=2, 
    n_digits=100, 
    batch_size=512,
    no_dupes=False,
    seed=0,
    device=None
):
    """
    Get validation-only dataset and dataloader for transformer model accuracy.
    
    Transformer models are trained on the 80% train split (default train_split=0.8),
    so accuracy must be measured on the held-out 20% validation split to properly
    measure generalization and avoid inflated metrics.
    
    Returns:
        (val_dataset, dataloader) where val_dataset is the held-out validation split
    """
```

**Key properties:**
- Returns validation split only (the 20% held-out data from `get_dataset()`)
- Default batch_size=512 (typical for model evaluation)
- Passes through `no_dupes` and other `get_dataset()` kwargs
- Self-documenting name makes the intent clear

## Scope of Changes

### Files to Update

**Core utilities (new):**
- `src/utils/nb_utils.py` — add the two helper functions

**Scripts (update to use helpers):**
- `scripts/train_sae.py`
- `scripts/compare_sae.py`
- `scripts/run_crossover_analysis.py`
- `scripts/nb_sae_feat_analysis.py`
- `scripts/special_latents_across_saes.py`
- `scripts/nb_model_interp.py` (if it evaluates model accuracy)

**Documentation:**
- `AGENTS.md` — add section documenting the standardization and when to use each helper
- Docstrings in helpers themselves

**Out of scope:**
- Archive scripts (frozen for reproducibility)
- Notebooks (live analysis, can be updated gradually)

## Migration Strategy

1. Add helpers to `src/utils/nb_utils.py` (non-breaking change)
2. Update scripts one-by-one in separate commits
3. Update AGENTS.md with new standards
4. Existing `ConcatDataset([train_ds, val_ds])` calls → `get_eval_dataset_sae()`
5. Existing `val_ds` or validation-only patterns → `get_eval_dataset_model()`

## Success Criteria

- All scripts use the appropriate helper (no manual dataset concatenation)
- AGENTS.md documents both patterns and when each is used
- New analysis scripts automatically follow the convention (because helpers are discoverable)
- No change to model accuracy or SAE metrics (validation results stay the same)

## Notes

- Device parameter is included but not used in the returned dataloader (it goes to data, not device) — for consistency with other nb_utils functions
- Both helpers should be documented inline with clear comments about why the dataset choice matters
- The helpers accept `**kwargs` to pass through to `get_dataset()` for future flexibility
