# Add MSE to SAE Sweep Tables and Pipeline

## Problem

The SAE sweep tables (BatchTopK, JumpReLU, Matryoshka) and the selected-SAEs table
currently report Loss Recovered, Dead %, and (in the comparison report) Explained
Variance — but not MSE (Mean Squared Error). MSE is a standard, interpretable
reconstruction quality metric and should be included.

MSE is already computed at runtime in `compare_sae.py::evaluate_sae()` but is not
carried through to the markdown report, and is therefore lost before table generation.
WandB runs also do not log MSE (`reconstruction_mse` is read by `loading.py` but
never written).

## Approach

**Approach A**: Back-calculate MSE from existing data for immediate tables,
plus fix the full pipeline for future runs.

Two workstreams:

1. **Immediate**: A standalone script (`scripts/backfill_mse.py`) that reads the
   existing `sae_comparison_*.md` report, computes `orig_var` from the base model,
   and back-calculates MSE for all 1,870 rows. Outputs updated LaTeX tables.

2. **Pipeline**: Modify `compare_sae.py`, `plot_sae_sweep.py`, and `train_sae.py`
   so future runs produce MSE natively.

## Detailed Design

### 1. Back-calculation script: `scripts/backfill_mse.py`

**Purpose**: One-shot script to produce updated LaTeX tables with MSE column
from existing data.

**Workflow**:
1. Load base model (`models/2layer_100dig_64d.pt`)
2. Collect SEP token activations (same as `compare_sae.py` L99-134)
3. Center activations: `sep_acts_centered = sep_acts - act_mean`
4. Compute `orig_var = (sep_acts_centered ** 2).mean().item()`
5. Parse existing report using extended `parse_report()` that also extracts `exp_var`
6. For each row: `mse = orig_var * (1 - exp_var)`
7. Aggregate by `(sae_type, l0, d_sae)`: compute `mse_mean`, `mse_std`
8. Generate LaTeX tables (one per SAE type + selected SAEs table)

**Output**: `.tex` files in `results/compare_sae/figures/<sae_type>/` matching
the user's existing table format, with MSE column added.

**Note**: Back-calculated MSE has ~4 significant figures due to the report storing
Exp Var to 4 decimal places. This is negligible for aggregated mean ± std tables.

### 2. Pipeline changes

#### 2a. `compare_sae.py` — Add MSE to markdown report

- In `generate_markdown_report()`:
  - Add `MSE` column to Summary Table header (after `Exp Var`)
  - Add `mse` value to each data row
  - Format as `{mse:.6f}` for full precision

No other changes needed — `evaluate_sae()` already computes and returns `mse`.

#### 2b. `plot_sae_sweep.py` — Parse, aggregate, and emit MSE

- `parse_report()`: Extract MSE column from the report (new column index)
- `aggregate()`: Add `mse_mean` / `mse_std` to the grouped aggregation
- `write_latex_table()`:
  - Add MSE column with header `MSE ($\downarrow$)`
  - Position: after `Loss Recovered`, before `Dead %`
  - Formatting: 4 decimal places, `$mean \pm std$`
  - Bold/underline: MSE is lower-is-better (like Dead %)
- `write_markdown_table()`: Same treatment for markdown output

#### 2c. `train_sae.py` — Log MSE to wandb

- In `_log_wandb_eval_metrics()`, after `compute_reconstruction_metrics()`:
  ```python
  wandb.summary["reconstruction_mse"] = recon["mse"]
  ```
- One-line change. `compare_sweep_runs()` in `loading.py` already reads
  `reconstruction_mse` from wandb summary, so no changes needed there.

### 3. LaTeX formatting

**Column specification**:
- Header: `MSE ($\downarrow$)`
- Position: After Loss Recovered, before Dead %
- Decimals: 4
- Format: `$0.0042 \pm 0.0003$`
- Scoring: Lower is better — bold for best in L0 block, underline+bold for global best

**Applies to**:
- Table 1 (BatchTopK sweep)
- Table 2 (JumpReLU sweep)
- Table 3 (Matryoshka sweep)
- Table 4 (Selected SAEs) — MSE in the Performance section

## Files Changed

| File | Change |
|------|--------|
| `scripts/backfill_mse.py` | **[NEW]** One-shot MSE back-calculation and LaTeX generation |
| `scripts/compare_sae.py` | Add MSE column to `generate_markdown_report()` |
| `scripts/plot_sae_sweep.py` | Parse MSE from report; aggregate; emit in LaTeX/markdown tables |
| `scripts/train_sae.py` | Log `reconstruction_mse` to wandb summary |

## Verification

1. Run `backfill_mse.py` — verify it produces valid LaTeX tables with MSE values
2. Spot-check: for a few SAEs, manually verify `MSE ≈ orig_var × (1 - exp_var)`
3. Run `plot_sae_sweep.py` on the existing report — verify it handles the MSE column gracefully (backward compat: old reports without MSE should still work)
4. Verify `train_sae.py` change is syntactically correct (no runtime test needed unless running a sweep)
5. Run existing tests: `.venv/bin/pytest tests/`
