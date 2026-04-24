# SAE Evaluation Methodology: All-Dataset vs Validation-Only

## Decision: Using All 10,000 Samples (train + val combined)

**Choice:** `all_dl` (train + validation combined, 10,000 samples)

**Rationale for this project:**

### Why We Use All-Dataset Evaluation

1. **Finite Exhaustive Task Space**
   - This task generates 100² = 10,000 unique input pairs
   - The train/val split (80/20) is arbitrary—there is no true "new" distribution
   - All possible inputs are represented exactly once in the full dataset

2. **Consistency with Published Metrics**
   - SAE comparison file (`sae_comparison_20260416_181925.md`) uses all 10,000 samples
   - Rankings and recommendations are calibrated against these full-dataset metrics
   - Using validation-only metrics would misalign with published benchmarks

3. **Mechanistic Analysis Requirements**
   - Interpretability studies need complete coverage across the task space
   - Special features, attention patterns, and activation phenomena are task-specific
   - Evaluating on 20% of inputs misses important structure

4. **Statistical Stability**
   - 10,000 samples → stable, low-variance metrics
   - 2,000 samples → higher noise, less reliable comparisons between models

### Trade-off: Data Leakage vs Task Completeness

**Potential concern:** Using training data in evaluation introduces "data leakage"
- **Mitigation:** This is acceptable because:
  - The SAE itself was trained on a different dataset split/model than what we're analyzing
  - The leakage is symmetric—all models evaluated on the same data
  - The task space is mathematically exhaustive (not truly "held-out")

### When to Use Validation-Only (`val_dl`)

Use validation-only metrics if:
- ✗ Publishing generalization claims to external audiences
- ✗ Comparing against algorithms trained on different data
- ✗ Claiming honest held-out performance

For this project, use all-dataset metrics unless explicitly doing held-out evaluation studies.

---

## Implementation

### Scripts Using All-Dataset Evaluation

**`scripts/nb_sae_feat_analysis.py`**
```python
# Collect activations on exhaustive dataset
sae_acts_all = collect_sae_activations(model, sae, all_dl, ...)

# Evaluate metrics on same exhaustive dataset
downstream = compute_sae_downstream_metrics(model, sae, all_dl, ...)
recon_metrics = compute_reconstruction_metrics(model, sae, all_dl, ...)
```

### Baseline Metrics (for reference)

On full dataset (10,000 samples):
- **Baseline CE:** 0.1115 (unpatched model accuracy)
- All SAE metrics reported are relative to this baseline

### Dataset Split Composition

```
Full dataset: 10,000 samples
├── Train: 8,000 samples (80%)
└── Val:   2,000 samples (20%)
```

All evaluation uses the full 10,000 for consistency.

---

## References

- SAE Comparison: `sae_comparison_20260416_181925.md`
- Final candidates: `SAE_FINAL_CANDIDATES_k3_d128_FILTERED.md`
- Script updated: `scripts/nb_sae_feat_analysis.py` (line 84: `val_dl` → `all_dl`)

