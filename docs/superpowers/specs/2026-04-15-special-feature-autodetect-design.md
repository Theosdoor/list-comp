# Special Feature Auto-Detection in Crossover Pipeline

**Date:** 2026-04-15  
**File:** `scripts/run_crossover_analysis.py`  
**Status:** Approved

---

## Problem

The crossover pipeline currently hardcodes feature index 30 as the default (via `--feature 30`). The "special" feature for a given SAE — the one with the highest correlation to attention difference (alpha_d1 − alpha_d2) — must be identified manually via `scripts/compare_sae.py`. This is a friction point: running the pipeline on a new SAE requires a separate manual step to find the right feature index.

---

## Goal

Insert automatic special-feature detection as a numbered pipeline step, so that:

1. The pipeline discovers which feature(s) to analyse on its own.
2. Results are saved per-feature in an organised directory structure.
3. A `special_features.md` summary is written alongside results for traceability.
4. The manual `--feature <int>` override is preserved for backward-compatibility.

---

## CLI Changes

Three arguments are modified or added in `parse_args()`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--feature` | `str` | `'auto'` | Feature index to analyse. `'auto'` triggers detection; an integer value skips detection and uses that index directly. |
| `--threshold` | `float` | `0.5` | Correlation threshold for `identify_special_features`. Features with \|corr\| > threshold are considered special. Only used when `--feature auto`. |
| `--max-features` | `int` | `2` | Maximum number of features to run the per-feature pipeline on (top N by \|corr\| descending). Only used when `--feature auto`. |

The startup header block prints all three values.

---

## Pipeline Structure

The pipeline is renumbered to 8 steps. Steps 1–4 are shared (run once). Steps 5–8 loop per feature.

```
[1/8] Load models
[2/8] Load dataset
[3/8] Collect SAE activations            (unchanged)
[4/8] Collect attention patterns         (NEW)
      + detect special features / resolve override

─── for each feature (up to --max-features) ───
[5/8] Find crossovers
[6/8] Identify swap zones
[7/8] Verify output swaps
[8/8] Generate report                    (if --report)
```

### Auto-detection path (`--feature auto`)

Step 4 proceeds as follows:

1. Call `collect_attention_patterns(model, all_dl, layer_idx=0, sep_idx=sep_idx, device=device)` to obtain `alpha_d1_all`, `alpha_d2_all`.
2. Call `identify_special_features(sae_acts_all, alpha_d1_all, alpha_d2_all, threshold=args.threshold)`.
3. Sort results by `|correlation|` descending.
4. If no features found: print warning and exit cleanly (no CSVs written).
5. Take the top `args.max_features` features for the loop. Keep all found features in the summary table (with a note if the cap truncated them).
6. Print the summary table to the log.
7. Write `special_features.md` to `results/xover/<sae_name>/`.

### Override path (`--feature <int>`)

- Steps 3 (attention collection) and 4 (detection) are skipped entirely.
- Feature list is `[int(args.feature)]`; the loop runs exactly once.
- No `special_features.md` is written.

---

## Directory Layout

```
results/xover/
└── <sae_name>/
    ├── special_features.md              ← SAE-level summary (auto mode only)
    └── <feat_idx>/
        ├── xovers_feat<N>.csv
        ├── swap_bounds_feat<N>.csv
        ├── swap_results_feat<N>.csv
        ├── failure_analysis_feat<N>.md  ← if --report
        └── plots/
```

`special_features.md` is at the SAE level (one directory above the feature dirs) because it describes the SAE as a whole, not a single feature run.

---

## `special_features.md` Format

```markdown
# Special Features — <sae_name>

Threshold: 0.5 | Found: 3 | Running top 2 (--max-features)

| Feature | Type | Correlation | Firing Rate |
|---------|------|-------------|-------------|
| 30      | d1_favoring | +0.821 | 0.4923 |
| 47      | d2_favoring | -0.763 | 0.3101 |
| 12      | d1_favoring | +0.612 | 0.2847 |
```

- Rows sorted by `|correlation|` descending.
- All found features are shown; a note clarifies if only top N were run.
- `firing_rate` = fraction of samples where feature activation > 0, computed from `sae_acts_all` (no extra pass needed).
- Same table is printed to stdout at detection time.

---

## Return Value of `run_pipeline`

`run_pipeline` returns a **list** of per-feature context dicts (one entry per feature run) rather than a single dict. Each dict includes the same keys as the current single dict, plus `feature_idx`.

`run_report` accepts a single per-feature context dict (unchanged interface, just called in a loop from `main`).

---

## No-Features-Found Behaviour

If auto-detection finds no features above `--threshold`:

```
WARNING: No special features found above threshold 0.5 for <sae_name>.
Consider lowering --threshold or inspecting the SAE with compare_sae.py.
Exiting.
```

Script exits with code 0 (not an error, just nothing to do). No output files are written.

---

## Functions Used

| Function | Source | Already exported? |
|----------|--------|-------------------|
| `collect_attention_patterns` | `src/sae/activation_collection.py` | Yes |
| `identify_special_features` | `src/sae/activation_collection.py` | Yes |

No new functions need to be added to the `src/sae` package.

---

## Out of Scope

- Changes to `compare_sae.py`.
- Changes to `src/sae/activation_collection.py`.
- Parallelising the per-feature loop.
- Threshold auto-tuning.
