# Code Review Report

**Date:** May 7, 2026  
**Scope:** Commits `59f62c3` to `0dfd53c`  
**Primary Script:** `scripts/nb_sae_feat_analysis.py`

## 1. Strengths

*   **Advanced Analysis Pipeline:** The repository features a highly sophisticated and automated pipeline for analyzing Sparse Autoencoders (SAEs). The integration of reconstruction metrics, downstream loss recovery, and mechanistic interpretation (correlation with attention) is state-of-the-art for this type of research.
*   **Robust Crossover Detection:** The steering logic in `src/sae/steering.py` is exceptionally robust. Utilizing a hybrid approach—analytical linear fits for predictable behavior (O1) and grid search + bisection for nonlinear behavior (O2)—ensures both precision and performance. The vectorized bisection implementation is a notable optimization for batch processing.
*   **Comprehensive Visualization:** The automated generation of heatmaps, boxplots, and scatter plots across multiple SAE types (BTK, JumpReLU, Matryoshka) allows for rapid comparative analysis. The recent improvements to `special_latents_across_saes.py` have significantly enhanced the clarity and organization of these results.
*   **Modular Architecture:** The refactoring of `sae_analysis.py` into specialized submodules (`hooks`, `steering`, `loading`, etc.) follows clean code principles and improves maintainability.

## 2. Issues

### Critical
*   **None identified.** The codebase is stable, well-tested, and demonstrates high technical integrity.

### Important
*   **Checkpoint Selection Logic Redundancy:** The `select_checkpoints` logic is implemented twice (once in `scripts/special_latents_across_saes.py` and once in `scripts/compare_sae.py`). This should be moved to a shared utility module (e.g., `src/sae/loading.py` or `src/utils/nb_utils.py`) to prevent logic drift.
*   **Hardcoded Scale Cap Inconsistency:** In `src/sae/steering.py`, the `LINEAR_FIT_SCALE_CAP` is set to `float('inf')`, but the comments explicitly state it should be a "soft upper cap" of `20.0`. This discrepancy could lead to accepting implausible extrapolations during O1 crossover detection.

### Minor
*   **Hardcoded Paths in Notebook Script:** `scripts/nb_sae_feat_analysis.py` contains hardcoded SAE and Model names. While acceptable for a notebook environment, adding command-line arguments (similar to `compare_sae.py`) would increase its utility as a diagnostic tool.
*   **Inconsistent `ConcatDataset` usage:** While most analysis scripts have moved to using `ConcatDataset([train_ds, val_ds])` for fuller evaluation, some utility functions might still default to validation only. A repo-wide standard for "evaluation dataset" would be beneficial.
*   **Missing Type Hints:** Several new functions in the scripts (e.g., `select_checkpoints`, `parse_d_sae_from_path`) lack type hints, which contrasts with the well-typed `src/` directory.

## 3. Symbol Peeling Analysis

The "Greedy Symbol Peeling" logic in `scripts/nb_sae_feat_analysis.py` (Cell 1b) is a significant analytical innovation. It effectively addresses the "disjunction of symbols" hypothesis for SAE latents.

*   **Logic Evaluation:** The greedy approach (picking the most frequent symbol in remaining activations) is sound. The use of a magnitude filter before peeling ensures that the analysis focuses on the latent's primary representational role rather than low-magnitude noise.
*   **Mathematical Correctness:** The formula for `expected` coverage (`k * (2 * N - 1) - k * (k - 1)`) correctly accounts for the combinatorics of bigram triggers in the list-copy task.
*   **Classification Depth:** By classifying latents as "X-symbol" detectors, the script provides a much higher resolution of interpretability than simple firing rate metrics.

## 4. Consistency with GEMINI.md

The recent changes are highly consistent with the mandates in `GEMINI.md`:
*   **SAE Loading:** Scripts correctly use `load_sae` and handle `act_mean` as required.
*   **Special Features:** The identification of "Special" features correctly uses the Pearson correlation with `alpha_d1 - alpha_d2` at the SEP token.
*   **Crossover Analysis:** The implementation of linear fits for O1 and grid search for O2 aligns perfectly with the research methodology described in the project overview.

## 5. Assessment

**Grade: Excellent**

The repository represents a high standard of research engineering. The transition from manual notebook analysis to a robust, automated pipeline is nearly complete. The "Greedy Symbol Peeling" and "Vectorized Bisection" are standout technical implementations. Addressing the minor redundancy in checkpoint selection and the scale cap inconsistency will further solidify the codebase.
