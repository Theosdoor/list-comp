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

**Recommended approach**: Back-calculate MSE from existing report data for the
paper tables, plus fix the full pipeline for future runs.

Alternatives considered:

- **Manual table edit**: fastest for the paper draft, but not reproducible and
  easy to desynchronise from the comparison report.
- **Full comparison re-run**: most direct way to get native MSE, but expensive
  and unnecessary because the existing report already stores explained variance
  and the checkpoints store the centering mean needed to reconstruct MSE.
- **Backfill + pipeline fix**: produces the immediate paper tables
  reproducibly, while preventing the same missing-column problem in future runs.

Target LaTeX tables:

- `tab:btk_sae_sweep` — BatchTopK sweep table
- `tab:jrelu_sae_sweep` — JumpReLU sweep table
- `tab:matryoshka_sae_sweep` — Matryoshka sweep table
- `tab:selected-saes` — selected high-performing checkpoints table

Two workstreams:

1. **Immediate**: A standalone script (`scripts/backfill_mse.py`) that reads the
   existing `sae_comparison_*.md` report, computes the original activation
   variance under the same centering convention as `compare_sae.py`, and
   back-calculates MSE for all rows in the selected report. Outputs updated
   replacement LaTeX for the four target tables above.

2. **Pipeline**: Modify `compare_sae.py`, `plot_sae_sweep.py`, and `train_sae.py`
   so future runs produce MSE natively.

## Detailed Design

### 1. Back-calculation script: `scripts/backfill_mse.py`

**Purpose**: One-shot script to produce updated LaTeX tables with MSE column
from existing data.

**Workflow**:
1. Load base model (`models/2layer_100dig_64d.pt`)
2. Collect SEP token activations (same as `compare_sae.py` L99-134)
3. Parse existing report using extended `parse_report()` that also extracts `exp_var`
4. Resolve each report row to the corresponding SAE checkpoint and its saved
   `act_mean`
5. Center activations with that checkpoint's saved mean:
   `sep_acts_centered = sep_acts - act_mean`
6. Compute row-specific `orig_var = (sep_acts_centered ** 2).mean().item()`
7. For each row: `mse = orig_var * (1 - exp_var)`
8. Aggregate by `(sae_type, l0, d_sae)`: compute `mse_mean`, `mse_std`
9. Generate LaTeX replacement snippets for the four target tables:
   - the three per-architecture sweep tables generated from aggregated report rows
   - the selected-SAEs table generated from an explicit selected-checkpoint
     mapping, because the visible table configuration columns do not include
     seed/checkpoint identity and may not uniquely identify a report row

**Output**: `.tex` files in `results/compare_sae/figures/<sae_type>/` matching
the existing sweep-table format, plus a selected-SAEs replacement snippet, all
with MSE columns added.

**Note**: Back-calculated MSE has ~4 significant figures due to the report storing
Exp Var to 4 decimal places. This is negligible for aggregated mean ± std tables.
If all selected checkpoints have identical saved `act_mean` tensors within
tolerance, the script may cache and reuse a single `orig_var`; otherwise it must
use the row-specific value.

The script should fail loudly if a report row cannot be matched to a checkpoint
for row-specific backfill. Missing or ambiguous selected-table rows should be
reported with the exact selected-checkpoint key that failed to match.

### 2. Pipeline changes

#### 2a. `compare_sae.py` — Add MSE to markdown report

- In `generate_markdown_report()`:
  - Add `MSE` column to Summary Table header (after `Exp Var`)
  - Add `mse` value to each data row
  - Format as `{mse:.6f}` for full precision

No other changes needed — `evaluate_sae()` already computes and returns `mse`.

#### 2b. `plot_sae_sweep.py` — Parse, aggregate, and emit MSE

- `parse_report()`: Build a header map from the summary table and extract
  columns by name, not by fixed index. This must support:
  - old reports without MSE
  - new reports with MSE inserted after `Exp Var`
  - `N Special` headers that include extra text such as `(mean|r|, thresh=0.5)`
- `aggregate()`: Add `mse_mean` / `mse_std` to the grouped aggregation
- `write_latex_table()`:
  - Add MSE column with header `MSE ($\downarrow$)`
  - Position: after `Loss Recovered`, before `Dead %`
  - Formatting: 4 decimal places, `$mean \pm std$`
  - Bold/underline: MSE is lower-is-better (like Dead %)
- `write_markdown_table()`: Same treatment for markdown output

Backward compatibility:

- If the parsed report contains native MSE, emit MSE columns directly.
- If the parsed report does not contain MSE and no backfilled MSE values are
  supplied, keep the current no-MSE table output rather than emitting empty MSE
  columns.
- `scripts/backfill_mse.py` is responsible for adding MSE to historical report
  data before calling the shared aggregation/table-writing helpers.

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

For `tab:selected-saes`, insert MSE in the Performance group after Loss Recovered
and before Dead %. The selected rows must be keyed by exact checkpoint name/path
or an explicit mapping from table row to report row. The visible configuration
columns (`d_sae`, target `L0`/`k`, learning rate, sparsity penalty, and
`n_groups`) are display columns, not a unique lookup key.

## Files Changed

| File | Change |
|------|--------|
| `scripts/backfill_mse.py` | **[NEW]** One-shot MSE back-calculation and LaTeX generation |
| `scripts/compare_sae.py` | Add MSE column to `generate_markdown_report()` |
| `scripts/plot_sae_sweep.py` | Parse MSE from report; aggregate; emit in LaTeX/markdown tables |
| `scripts/train_sae.py` | Log `reconstruction_mse` to wandb summary |
| `tests/test_plot_sae_sweep.py` | Cover old/new report parsing and MSE formatting |

## Verification

1. Run `backfill_mse.py` — verify it produces valid LaTeX tables with MSE values
2. Spot-check: for a few SAEs, manually verify `MSE ≈ orig_var × (1 - exp_var)`
3. Run `plot_sae_sweep.py` on the existing report — verify old reports without
   MSE still produce the existing table shape
4. Add tests for:
   - parsing an old report without MSE
   - parsing a new report with MSE after `Exp Var`
   - preserving the `N Special` parse when the header includes extra text
   - lower-is-better MSE bold/underline formatting, including rounded ties
5. Verify `train_sae.py` change is syntactically correct (no runtime test needed unless running a sweep)
6. Run existing tests: `.venv/bin/pytest tests/`
