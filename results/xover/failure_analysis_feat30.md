# Failure Reason Analysis — Feature 30

Pipeline: `get_xovers_df` → `get_output_swap_bounds`. Correctness is the base model accuracy at scale=1.0 (unsteered), classified per-position: **both_correct** (o1=d1 and o2=d2), **partial** (one position correct), **both_wrong**.

## Summary

| failure_reason | both_correct | partial | both_wrong | total | % of all |
|---|---|---|---|---|---|
| `success` | 5495 | 524 | 21 | 6040 | 60.4% |
| `feat_zero` | 2698 | 236 | 12 | 2946 | 29.5% |
| `d1_eq_d2` | 83 | 0 | 0 | 83 | 0.8% |
| `o1_negative_scale` | 243 | 276 | 8 | 527 | 5.3% |
| `o1_extrapolated` | 1 | 18 | 0 | 19 | 0.2% |
| `no_o2_crossover` | 13 | 29 | 0 | 42 | 0.4% |
| `no_o2_crossover_in_bounds` | 122 | 75 | 1 | 198 | 2.0% |
| `no_overlapping_dominance` | 10 | 30 | 2 | 42 | 0.4% |
| `o1_never_predicts_d2` | 3 | 84 | 1 | 88 | 0.9% |
| `o2_never_predicts_d1` | 2 | 11 | 1 | 14 | 0.1% |
| `invalid_bounds` | 0 | 1 | 0 | 1 | 0.0% |
| **TOTAL** | **8670** | **1284** | **46** | **10000** | 100% |

## Per-Reason Breakdown

### `success` (6040 samples)

**Correctness:** 5495 both_correct / 524 partial / 21 both_wrong

The pipeline found a valid swap zone: both o1 and o2 crossovers resolved correctly and argmax dominance confirmed a contiguous scale window where o1 predicts d2 and o2 predicts d1.

**Examples** (up to 5 of 6040):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 28 | 90 | 2.368 | 0.395 | 0.399, 3.942 | 0.000 | 0.350 | both_correct |
| 93 | 99 | 5.049 | 0.626 | 0.220, 2.073 | 0.000 | 0.100 | both_correct |
| 19 | 17 | 1.157 | 1.753 | 1.647, 8.523 | 1.800 | 5.600 | both_correct |
| 49 | 19 | 1.845 | 2.159 | 0.900, 4.259 | 2.200 | 2.250 | partial |
| 17 | 29 | 4.623 | 0.652 | 0.188, 2.090 | 0.000 | 0.050 | both_correct |

### `feat_zero` (2946 samples)

**Correctness:** 2698 both_correct / 236 partial / 12 both_wrong

The feature has zero activation on this input, so steering it does nothing. This is normal; the feature simply doesn't fire for every digit pair.

**Examples** (up to 5 of 2946):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 75 | 32 | 0.000 | — | — | — | — | both_correct |
| 48 | 69 | 0.000 | — | — | — | — | both_correct |
| 7 | 84 | 0.000 | — | — | — | — | both_correct |
| 48 | 40 | 0.000 | — | — | — | — | both_correct |
| 63 | 1 | 0.000 | — | — | — | — | both_correct |

### `d1_eq_d2` (83 samples)

**Correctness:** 83 both_correct / 0 partial / 0 both_wrong

Both input digits are the same (d1 == d2). The crossover framework is degenerate here because the 'swap' (d2, d1) is identical to the normal output (d1, d2).

**Examples** (up to 5 of 83):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 77 | 77 | 6.582 | — | — | — | — | both_correct |
| 78 | 78 | 2.884 | — | — | — | — | both_correct |
| 70 | 70 | 15.379 | — | — | — | — | both_correct |
| 15 | 15 | 2.701 | — | — | — | — | both_correct |
| 39 | 39 | 0.716 | — | — | — | — | both_correct |

### `o1_negative_scale` (527 samples)

**Correctness:** 243 both_correct / 276 partial / 8 both_wrong

The analytical o1 crossover (linear fit) falls at a negative scale. This means d2 already beats d1 at o1 even at scale=0 — suppressing the feature swaps the output, not amplifying it. **Note:** with the updated `_find_o1_crossover_linear` this case is now returned as a valid crossover rather than a failure; these rows will process on re-run.

**Examples** (up to 5 of 527):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 60 | 44 | 1.030 | — | 0.917, 7.212 | — | — | both_correct |
| 9 | 81 | 1.185 | — | 0.910, 8.633 | — | — | both_correct |
| 42 | 33 | 0.791 | — | 0.598, 8.987 | — | — | both_correct |
| 39 | 35 | 0.531 | — | 0.540 | — | — | partial |
| 55 | 76 | 1.074 | — | — | — | — | partial |

### `o1_extrapolated` (19 samples)

**Correctness:** 1 both_correct / 18 partial / 0 both_wrong

The analytical o1 crossover is beyond scale 20 (2× the grid ceiling). The linear model is extrapolating far outside tested territory, so we flag rather than trust the value.

**Examples** (up to 5 of 19):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 89 | 60 | 1.013 | — | 3.487 | — | — | partial |
| 38 | 39 | 1.628 | — | 1.577 | — | — | partial |
| 30 | 3 | 2.169 | — | 0.976 | — | — | both_correct |
| 44 | 39 | 0.769 | — | 1.622 | — | — | partial |
| 61 | 63 | 1.193 | — | — | — | — | partial |

### `no_o2_crossover` (42 samples)

**Correctness:** 13 both_correct / 29 partial / 0 both_wrong

No sign change in the d1−d2 logit diff at o2 across the whole scale grid. Either d1 was already beating d2 at o2 throughout, or d2 was always dominant. The pipeline has a fallback that accepts this if argmax_o2 == d1 somewhere in the o1-constrained window — this failure means even that fallback found nothing.

**Examples** (up to 5 of 42):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 35 | 88 | 0.595 | 0.777 | — | — | — | both_correct |
| 18 | 86 | 4.794 | 2.651 | — | — | — | both_correct |
| 35 | 44 | 0.483 | 12.467 | — | — | — | partial |
| 30 | 21 | 2.207 | 12.297 | — | — | — | partial |
| 15 | 11 | 3.158 | 0.143 | — | — | — | partial |

### `no_o2_crossover_in_bounds` (198 samples)

**Correctness:** 122 both_correct / 75 partial / 1 both_wrong

An o2 crossover exists, but outside the scale window constrained by the o1 crossover. The argmax fallback also found no grid point where argmax_o2 == d1 within the window.

**Examples** (up to 5 of 198):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 47 | 31 | 3.737 | 0.098 | 0.751, 4.146 | — | — | partial |
| 69 | 93 | 3.000 | 0.202 | 0.907, 3.245 | — | — | both_correct |
| 34 | 11 | 5.226 | 0.529 | 0.830, 2.526 | — | — | both_correct |
| 68 | 0 | 4.506 | 0.382 | 0.854, 3.437 | — | — | both_correct |
| 0 | 78 | 3.191 | 0.489 | 1.020, 4.970 | — | — | partial |

### `no_overlapping_dominance` (42 samples)

**Correctness:** 10 both_correct / 30 partial / 2 both_wrong

The argmax dominance ranges for o1 (predicts d2) and o2 (predicts d1) never overlap. Typically a third digit takes over the argmax in the middle of the intended swap window, breaking the required simultaneous condition.

**Examples** (up to 5 of 42):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 21 | 82 | 1.807 | 1.889 | 1.003 | — | — | both_correct |
| 30 | 11 | 2.695 | 1.524 | 0.802, 4.841 | — | — | partial |
| 98 | 1 | 2.255 | 1.812 | 0.815, 5.690 | — | — | partial |
| 18 | 74 | 6.492 | 0.652 | 0.142, 1.907 | — | — | both_correct |
| 77 | 62 | 6.756 | 0.672 | 0.099, 1.943 | — | — | partial |

### `o1_never_predicts_d2` (88 samples)

**Correctness:** 3 both_correct / 84 partial / 1 both_wrong

Even though a crossover scale was found for o1, argmax_o1 never actually equals d2 on the coarse grid — a third digit steals the top logit before d2 can take over.

**Examples** (up to 5 of 88):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 14 | 58 | 1.206 | 6.798 | 0.104, 8.271 | — | — | partial |
| 6 | 17 | 1.523 | 8.079 | 0.767 | — | — | partial |
| 56 | 72 | 2.975 | 1.961 | 0.821, 3.365 | — | — | partial |
| 6 | 28 | 1.465 | 4.373 | 0.531 | — | — | partial |
| 21 | 28 | 1.981 | 6.142 | 0.722 | — | — | partial |

### `o2_never_predicts_d1` (14 samples)

**Correctness:** 2 both_correct / 11 partial / 1 both_wrong

No grid point has argmax_o2 == d1. A third digit is always dominant at o2, preventing the required swapped output.

**Examples** (up to 5 of 14):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 23 | 95 | 2.539 | 4.159 | 0.006, 7.679 | — | — | partial |
| 91 | 29 | 6.442 | 0.637 | 0.111, 1.931 | — | — | partial |
| 91 | 34 | 5.510 | 0.644 | 0.095, 0.684 | — | — | partial |
| 32 | 3 | 2.099 | 6.452 | 0.205 | — | — | partial |
| 49 | 43 | 1.636 | 2.657 | 0.189, 8.948 | — | — | partial |

### `invalid_bounds` (1 samples)

**Correctness:** 0 both_correct / 1 partial / 0 both_wrong

After processing all crossovers, lower_bound > upper_bound. Rare edge case where lb and ub crossovers from different positions are inconsistent.

**Examples** (up to 5 of 1):

| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |
|---|---|---|---|---|---|---|---|
| 75 | 29 | 2.600 | 3.060 | 0.738, 3.028 | — | — | partial |
