# Crossover Pipeline Performance Spec

Good. I now have everything I need. On the dataset question: with `no_dupes=False`, `itertools.product` produces exactly one row per (d1, d2) pair, and train/val is a partition, so the combined dataset has unique pairs. Building a dict lookup is safe. On DRY for vectorised bisection: I'll give you my recommendation in the spec below.


## Scope

Six concrete changes. Changes 1–3 are independent and low-risk. Changes 4–5 are coupled (orig outputs). Change 6 (vectorised bisection) is the most invasive and should be implemented last.

---

## Change 1 — feat_zero fast-path in `get_xovers_df`

**File:** `src/sae/steering.py`  
**Function:** `_process_crossover_batch`

**Problem:** Every sample in each batch — including those where `feat_orig == 0` — runs through 201 scale forward passes before the `feat_orig == 0` check fires inside `_analyze_single_sample_crossovers`. For sparse features, potentially the majority of the dataset does this work for nothing.

**Change:** Before calling `_run_batched_steering`, partition the batch by `feat_zero_mask = (batch_feat_orig == 0)`.

- For feat_zero samples: immediately produce `_empty_result('feat_zero', ...)` without running the grid. Pre-populate `orig_o1` and `orig_o2` as `None` (see Change 5 for why).
- For non-feat_zero samples: extract the subset (`batch_inputs[~feat_zero_mask]`, `batch_z_orig[~feat_zero_mask]`), run `_run_batched_steering` on that subset only, then run `_analyze_single_sample_crossovers` for each.
- Reassemble full results in original sample order before returning.

`_empty_result` should be updated to accept `scale_factors` so it can populate `scales`, `argmax_o1`, `argmax_o2` as empty lists consistently with the non-fast-path rows. Currently `_empty_result` is a closure inside `_analyze_single_sample_crossovers`; extract it to module level so it can be called from `_process_crossover_batch` too.

**Note:** `_analyze_single_sample_crossovers` already handles `feat_orig == 0` via its own `_empty_result` call. After this change, that branch becomes unreachable but can be left as a guard.

---

## Change 2 — Add `feat_zero` to `SUMMARY_ONLY`

**File:** `src/sae/reporting.py`

**Change:** One line:

```python
SUMMARY_ONLY = {"dead_latent", "feat_zero"}
```

`feat_zero` currently falls through to `generate_example_visuals`, which calls `feature_steering_experiment` on inputs where the feature never fires. The steering plots are flat and meaningless. The summary table already shows count and correctness via the merge with `correctness_df`, which is all that's needed.

---

## Change 3 — Move dead latent reclassification upstream

**Files:** `scripts/run_crossover_analysis.py`, `src/sae/reporting.py`

**Problem:** The reclassification of `feat_zero → dead_latent` happens in `run_report` after reading CSVs. CSVs on disk remain inconsistent.

**Change in `run_pipeline`** (run_crossover_analysis.py): After `get_xovers_df` returns and before saving to CSV, check:

```python
is_dead_latent = (sae_acts_all[:, feature_idx] > 0).sum() == 0
if is_dead_latent:
    xovers_df = xovers_df.copy()
    xovers_df.loc[
        xovers_df['o1_failure_reason'] == 'feat_zero', 'o1_failure_reason'
    ] = 'dead_latent'
```

`_determine_swap_bounds_for_sample` already propagates `o1_failure_reason` directly to `failure_reason` in swap_bounds_df, so this flows through automatically to the downstream CSV.

**Change in `run_report`** (run_crossover_analysis.py): Remove the reclassification block (the `if (merged["feat_orig"].fillna(0) > 0).sum() == 0` block).

---

## Change 4 — `(d1, d2) → idx` lookup dict

**File:** `src/sae/steering.py`  
**Functions:** `swap_outputs`, `analyze_feature_crossovers`

**Problem:** `_find_input_index` does an O(n) tensor scan every call. `swap_outputs` calls it once per valid swap row; `analyze_feature_crossovers` calls it once per test case.

**Add a helper:**

```python
def _build_index_lookup(d1_all, d2_all) -> dict:
    """Build a {(d1, d2): index} dict for O(1) lookup.
    Safe when all (d1, d2) pairs are unique (guaranteed by full-enumeration dataset)."""
    return {
        (d1_all[i].item(), d2_all[i].item()): i
        for i in range(len(d1_all))
    }
```

**In `swap_outputs`:** Call `_build_index_lookup` once before the loop. Pass the resulting dict into `_verify_single_swap` (add a `idx_lookup` parameter). Replace the `_find_input_index` call with `idx_lookup[(d1_val, d2_val)]`.

**In `analyze_feature_crossovers`:** Same — build once, pass into `_analyze_single_result_crossovers`.

`_find_input_index` can be kept as a private utility (it may be useful in notebooks for one-off lookups) but should not be called in any hot loop.

---

## Change 5 — Store `orig_o1`, `orig_o2` in `xovers_df`; eliminate redundant scale=1.0 forward pass in `swap_outputs`

**Files:** `src/sae/steering.py`, `src/sae/reporting.py`

**Problem:** `_verify_single_swap` runs the model at scale=1.0 to get "original" outputs. Scale=1.0 is index 20 in the coarse grid (`np.linspace(0.0, 10.0, 201)`), so this result is already computed and thrown away.

**In `_analyze_single_sample_crossovers`** (steering.py): After computing `argmax_o1` and `argmax_o2` from the grid:

```python
scale_1_idx = int(np.argmin(np.abs(np.asarray(scale_factors) - 1.0)))
orig_o1 = int(argmax_o1[scale_1_idx])
orig_o2 = int(argmax_o2[scale_1_idx])
```

Add `orig_o1` and `orig_o2` to the returned dict. Add `orig_o1: None, orig_o2: None` to `_empty_result` (feat_zero and d1_eq_d2 cases have no grid, so None is correct).

**In `get_output_swap_bounds` / `_determine_swap_bounds_for_sample`** (steering.py): Pass `orig_o1` and `orig_o2` through from the xovers_df row into the result dict. They are not used for bound computation — they just need to survive into `swap_bounds_df`.

Update the `get_output_swap_bounds` docstring to note the two new passthrough columns.

**In `_verify_single_swap`** (steering.py): Read `orig_o1` and `orig_o2` from the row if present and non-null:

```python
if pd.notna(row.get('orig_o1')):
    orig_o1 = int(row['orig_o1'])
    orig_o2 = int(row['orig_o2'])
else:
    # fallback for CSVs produced before this change
    orig_logits = _run_model_with_scaled_feature(...)
    orig_o1 = orig_logits[0, OUTPUT_POS_O1, :n_digits].argmax().item()
    orig_o2 = orig_logits[0, OUTPUT_POS_O2, :n_digits].argmax().item()
```

**In `build_merged`** (reporting.py): Add `orig_o1`, `orig_o2` to `xovers_slim` column list if present, so they're available in the merged report df.

---

## Change 6 — Vectorised bisection for o2 crossovers

**File:** `src/sae/steering.py`

**On DRY:** Recommended approach is two separate implementations. `find_exact_crossover_bisection` (scalar, public) stays untouched — it's used by `analyze_feature_crossovers` which runs on ~5 examples where batching adds overhead and complexity for no meaningful gain. The new `_run_vectorised_bisection` (batched, internal) is used only by the pipeline path. The shared logic is minimal enough (bracket update rule) that duplication is acceptable. If you want DRY, `find_exact_crossover_bisection` can delegate to `_run_vectorised_bisection` with a single-task list — but profile first, the overhead may not be worth it.

**New function:** `_run_vectorised_bisection`

```python
def _run_vectorised_bisection(
    tasks: list[dict],
    model, sae, act_mean, feature_idx,
    layer_idx, sep_idx, n_digits, device,
    tol=DEFAULT_BISECTION_TOL,
    max_iter=DEFAULT_BISECTION_MAX_ITER,
) -> list[float]:
```

Each task dict contains: `inputs_i` [1, seq_len], `z_orig` [d_sae], `feat_orig` float, `d1_val` int, `d2_val` int, `scale_low` float, `scale_high` float, `output_pos` int.

Returns a list of exact crossover scales, one per task, in input order.

**Algorithm:**

Step 0 — pre-compute `diff_low` for all tasks in one batched forward pass (stack all `inputs_i`, set `z_scaled[:, feature_idx] = feat_orig * scale_low` per task). Extract `logit[d1] - logit[d2]` at `output_pos` for each task → `diff_lows: list[float]`.

Step 1 — bisection loop (up to `max_iter`):
- `active`: boolean mask, initially all True
- Compute `mids[i] = (lows[i] + highs[i]) / 2` for all active tasks
- Mark as converged (inactive) any task where `highs[i] - lows[i] < tol`
- If no active tasks remain, break
- Build batch from active tasks: stack `inputs_i`, set `z_scaled[j, feature_idx] = feat_orig_j * mids[j]` for each
- One batched forward pass → `diff_mids[j]` for each active task
- Update brackets: if `sign(diff_mids[j]) == sign(diff_lows[j])`, then `lows[j] = mids[j]`, `diff_lows[j] = diff_mids[j]`; else `highs[j] = mids[j]`

Return `[(lows[i] + highs[i]) / 2 for i in range(len(tasks))]`.

Note: this uses one forward pass per iteration rather than two (the standard bisection optimisation — when `low` updates to `mid`, `diff_low` for the next iteration is already known as `diff_mid`).

**Restructure `_process_crossover_batch`:**

After the grid phase and the feat_zero fast-path (Change 1), the non-feat_zero samples need their o2 sign changes collected first, then bisected together.

Split `_find_crossovers_for_position` into two functions:
1. `_collect_o2_sign_changes(logits, d1_val, d2_val, scale_factors)` → list of `(scale_low, scale_high, crossover_idx)`. No model calls. Pure numpy.
2. Remove bisection from `_find_crossovers_for_position`. After batched bisection produces exact scales, bound types are computed from `_determine_bound_type_from_diff` as before.

New flow in `_process_crossover_batch` for non-feat_zero samples:

1. Run grid → `batch_logits_o1`, `batch_logits_o2`
2. For each sample, collect o2 sign changes and build bisection tasks (include `inputs_i = batch_inputs[i]`, `z_orig = batch_z_orig[i]`, etc.)
3. Call `_run_vectorised_bisection(all_tasks, ...)` → one list of exact scales
4. Distribute exact scales back to per-sample crossover lists
5. For each sample, compute bound types, run `_find_o1_crossover_linear`, assemble result dict

`_analyze_single_sample_crossovers` in its current form is effectively replaced by this restructured flow. It can be removed or reduced to a thin assembler that takes pre-computed o2 results.

**`analyze_feature_crossovers` is unchanged** — it retains its own call to `find_exact_crossover_bisection` via `_find_and_analyze_crossovers`.

---

## Implementation order

1 → 2 → 3 (independent, low-risk, do these first and verify the report still renders correctly)  
4 (independent, do after 1–3)  
5 (depends on grid being run, so after 1; before 6)  
6 (last — most invasive, requires 1 to be done first so the fast-path is already reducing the task count)