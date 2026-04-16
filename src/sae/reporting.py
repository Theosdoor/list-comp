"""
Failure-reason reporting for the crossover analysis pipeline.

Provides utilities for:
  - Labelling each input with original-model correctness
  - Merging crossover / swap-bounds DataFrames
  - Generating per-reason steering visuals
  - Rendering a Markdown report
"""
import json
import re

import numpy as np
import pandas as pd
import torch

# ── Constants ──────────────────────────────────────────────────────────────────

STEERING_SCALE_RANGE = [-1.0, 5.0]
STEERING_STEP = 0.05
N_EXAMPLES = 5

FAILURE_DESCRIPTIONS = {
    "success": (
        "The pipeline found a valid swap zone: both o1 and o2 crossovers resolved correctly "
        "and argmax dominance confirmed a contiguous scale window where o1 predicts d2 and o2 "
        "predicts d1."
    ),
    "dead_latent": (
        "The feature has zero activation on every input in the dataset — it is a dead latent "
        "that never learned to represent anything. Steering it has no effect on any input."
    ),
    "feat_zero": (
        "The feature has zero activation on this input, so steering it does nothing. "
        "This is normal; the feature simply doesn't fire for every digit pair."
    ),
    "d1_eq_d2": (
        "Both input digits are the same (d1 == d2). The crossover framework is degenerate here "
        "because the 'swap' (d2, d1) is identical to the normal output (d1, d2)."
    ),
    "o1_negative_scale": (
        "The analytical o1 crossover (linear fit) falls at a negative scale. "
        "This means d2 already beats d1 at o1 even at scale=0 — suppressing the feature "
        "swaps the output, not amplifying it. "
        "**Note:** with the updated `_find_o1_crossover_linear` this case is now returned as a "
        "valid crossover rather than a failure; these rows will process on re-run."
    ),
    "o1_extrapolated": (
        "The analytical o1 crossover is beyond scale 20 (2x the grid ceiling). "
        "The linear model is extrapolating far outside tested territory, so we flag "
        "rather than trust the value."
    ),
    "no_o2_crossover": (
        "No sign change in the d1-d2 logit diff at o2 across the whole scale grid. "
        "Either d1 was already beating d2 at o2 throughout, or d2 was always dominant. "
        "The pipeline has a fallback that accepts this if argmax_o2 == d1 somewhere in "
        "the o1-constrained window — this failure means even that fallback found nothing."
    ),
    "no_o2_crossover_in_bounds": (
        "An o2 crossover exists, but outside the scale window constrained by the o1 crossover. "
        "The argmax fallback also found no grid point where argmax_o2 == d1 within the window."
    ),
    "no_overlapping_dominance": (
        "The argmax dominance ranges for o1 (predicts d2) and o2 (predicts d1) never overlap. "
        "Typically a third digit takes over the argmax in the middle of the intended swap window, "
        "breaking the required simultaneous condition."
    ),
    "o1_never_predicts_d2": (
        "Even though a crossover scale was found for o1, argmax_o1 never actually equals d2 "
        "on the coarse grid — a third digit steals the top logit before d2 can take over."
    ),
    "o2_never_predicts_d1": (
        "No grid point has argmax_o2 == d1. A third digit is always dominant at o2, "
        "preventing the required swapped output."
    ),
    "invalid_bounds": (
        "After processing all crossovers, lower_bound > upper_bound. "
        "Rare edge case where lb and ub crossovers from different positions are inconsistent."
    ),
    "no_o1_crossover": (
        "o1_failure_reason was None (linear fit succeeded) but o1_crossovers is empty. "
        "Should not normally be reached; indicates an unexpected code path."
    ),
}

FAILURE_ORDER = [
    "success",
    "dead_latent",
    "feat_zero",
    "d1_eq_d2",
    "o1_negative_scale",
    "o1_extrapolated",
    "no_o2_crossover",
    "no_o2_crossover_in_bounds",
    "no_overlapping_dominance",
    "o1_never_predicts_d2",
    "o2_never_predicts_d1",
    "invalid_bounds",
    "no_o1_crossover",
]

# Reasons that appear only in the summary table — no breakdown section or visuals.
SUMMARY_ONLY = {"dead_latent"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_list_field(field):
    """Parse a serialised list field from a CSV column into a Python list."""
    if isinstance(field, (list, np.ndarray)):
        return list(field)
    if not isinstance(field, str):
        return field
    # Convert Python-style list repr to JSON-compatible form and parse
    cleaned = re.sub(r"\bnan\b", "null", field)
    cleaned = cleaned.replace("'", '"')
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return field


def _ordered_reasons(df: pd.DataFrame):
    """Return failure_reason values in canonical order, unknowns appended."""
    present = df["failure_reason"].value_counts().index.tolist()
    ordered = [r for r in FAILURE_ORDER if r in present]
    ordered += [r for r in present if r not in FAILURE_ORDER]
    return ordered


# ── Correctness labelling ──────────────────────────────────────────────────────

def get_original_correctness(model, all_dl, n_digits: int, list_len: int, device) -> pd.DataFrame:
    """
    Run the original transformer (no SAE hook) on all inputs and return a
    DataFrame with columns [d1, d2, correctness] where correctness is one of
    'both_correct', 'partial', 'both_wrong'.
    """
    rows = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in all_dl:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits[:, list_len + 1:, :n_digits].argmax(-1)
            tgt = targets[:, list_len + 1:]
            d1s = targets[:, 0].tolist()
            d2s = targets[:, 1].tolist()
            o1c = (preds[:, 0] == tgt[:, 0].to(device)).tolist()
            o2c = (preds[:, 1] == tgt[:, 1].to(device)).tolist()
            for d1, d2, c1, c2 in zip(d1s, d2s, o1c, o2c):
                if c1 and c2:
                    correctness = "both_correct"
                elif c1 or c2:
                    correctness = "partial"
                else:
                    correctness = "both_wrong"
                rows.append({"d1": int(d1), "d2": int(d2), "correctness": correctness})
    return pd.DataFrame(rows)


def build_merged(xovers_df: pd.DataFrame, bounds_df: pd.DataFrame,
                 model, all_dl, n_digits: int, list_len: int, device) -> pd.DataFrame:
    """Merge swap bounds (failure reasons) with xovers data and correctness labels."""
    print("Running original model inference for ground-truth correctness...")
    correctness_df = get_original_correctness(model, all_dl, n_digits, list_len, device)

    bounds = bounds_df.copy()
    bounds["failure_reason"] = bounds["failure_reason"].fillna("success")

    merged = bounds.merge(correctness_df, on=["d1", "d2"], how="left")
    xovers_slim = xovers_df[[
        "d1", "d2", "feat_orig", "o1_crossovers", "o2_crossovers",
        "o1_bound_types", "o2_bound_types", "n_o1_xover", "n_o2_xover",
        "o1_failure_reason",
    ]].copy()
    return merged.merge(xovers_slim, on=["d1", "d2"], how="left")


# ── Visual generation ──────────────────────────────────────────────────────────

def _format_crossover_df(df: pd.DataFrame) -> str:
    lines = [
        "| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | o1_failure_reason | o1_swapped | o2_swapped |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        def _fmt_list(v):
            lst = parse_list_field(v) if not isinstance(v, list) else v
            if not lst:
                return "—"
            try:
                return ", ".join(f"{x:.3f}" for x in lst)
            except (TypeError, ValueError):
                return str(lst)

        feat = row.get("feat_orig", "")
        feat_str = f"{feat:.4f}" if isinstance(feat, float) and not pd.isna(feat) else str(feat)
        reason = row.get("o1_failure_reason", "")
        reason_str = str(reason) if (isinstance(reason, str) and reason) else "—"
        o1_sw = parse_list_field(row.get("o1_swapped", [])) if not isinstance(row.get("o1_swapped"), list) else row.get("o1_swapped", [])
        o2_sw = parse_list_field(row.get("o2_swapped", [])) if not isinstance(row.get("o2_swapped"), list) else row.get("o2_swapped", [])
        lines.append(
            f"| {int(row['d1'])} | {int(row['d2'])} | {feat_str} "
            f"| {_fmt_list(row.get('o1_crossovers', []))} "
            f"| {_fmt_list(row.get('o2_crossovers', []))} "
            f"| {reason_str} | {o1_sw} | {o2_sw} |"
        )
    return "\n".join(lines)


def generate_example_visuals(
    group: pd.DataFrame,
    reason: str,
    model,
    sae,
    act_mean,
    d1_all,
    d2_all,
    sae_acts_all,
    all_ds,
    feature_idx: int,
    plots_dir,
    n_digits: int,
    list_len: int,
):
    """
    For up to N_EXAMPLES rows of `group`, run feature_steering_experiment and
    analyze_feature_crossovers. Saves a steering plot and returns
    (plot_rel_path, crossover_md) where paths are relative to the report location.
    """
    from src.sae.steering import feature_steering_experiment, analyze_feature_crossovers

    if len(group) == 0:
        return None, None

    sample = group.head(N_EXAMPLES)
    test_pairs = [(int(r["d1"]), int(r["d2"])) for _, r in sample.iterrows()]

    plots_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = reason.replace("/", "_")
    plot_filename = f"steering_feat{feature_idx}_{safe_reason}.png"
    plot_path = plots_dir / plot_filename
    plot_rel = f"plots/{plot_filename}"

    steering_results = feature_steering_experiment(
        model, sae, act_mean, feature_idx,
        d1_all, d2_all, sae_acts_all, all_ds,
        layer_idx=0, sep_idx=list_len, n_digits=n_digits,
        scale_range=STEERING_SCALE_RANGE,
        sample_step_size=STEERING_STEP,
        test_pairs=test_pairs,
        plot=True,
        save_path=str(plot_path),
    )

    if not steering_results:
        return None, None

    xover_df = analyze_feature_crossovers(
        steering_results, model, sae, act_mean, feature_idx,
        d1_all, d2_all, sae_acts_all, all_ds,
        layer_idx=0, sep_idx=list_len, n_digits=n_digits,
        verbose=False,
    )

    return plot_rel, _format_crossover_df(xover_df)


# ── Markdown rendering ─────────────────────────────────────────────────────────

def _correctness_row(group: pd.DataFrame):
    counts = group["correctness"].value_counts()
    bc = counts.get("both_correct", 0)
    pa = counts.get("partial", 0)
    bw = counts.get("both_wrong", 0)
    return bc, pa, bw, len(group)


def _fmt_example(row) -> str:
    o1x = parse_list_field(row.get("o1_crossovers", []))
    o2x = parse_list_field(row.get("o2_crossovers", []))
    o1x_str = ", ".join(f"{v:.3f}" for v in o1x) if o1x else "—"
    o2x_str = ", ".join(f"{v:.3f}" for v in o2x) if o2x else "—"
    feat = row.get("feat_orig", "")
    feat_str = f"{feat:.3f}" if isinstance(feat, (float, int)) and not pd.isna(feat) else str(feat)
    lb = row.get("lower_bound", "")
    ub = row.get("upper_bound", "")
    lb_str = f"{lb:.3f}" if (isinstance(lb, float) and not pd.isna(lb)) else "—"
    ub_str = f"{ub:.3f}" if (isinstance(ub, float) and not pd.isna(ub)) else "—"
    corr = row.get("correctness", "")
    return (
        f"| {int(row['d1'])} | {int(row['d2'])} | {feat_str} "
        f"| {o1x_str} | {o2x_str} | {lb_str} | {ub_str} | {corr} |"
    )


def _examples_table(group: pd.DataFrame, n: int = 5) -> str:
    sample = group.head(n)
    lines = [
        "| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in sample.iterrows():
        lines.append(_fmt_example(row))
    return "\n".join(lines)


def generate_markdown(merged: pd.DataFrame, feature_idx: int, visuals=None) -> str:
    """
    Render the full failure-reason markdown report.

    Args:
        merged:      DataFrame with failure_reason and correctness columns.
        feature_idx: SAE feature index (used in the title).
        visuals:     Optional dict mapping reason -> (plot_rel_path, crossover_md_str).
    """
    total = len(merged)
    ordered = _ordered_reasons(merged)
    lines = []

    lines.append(f"# Failure Reason Analysis — Feature {feature_idx}")
    lines.append("")
    lines.append(
        "Pipeline: `get_xovers_df` -> `get_output_swap_bounds`. "
        "Correctness is the original model accuracy (no SAE, no steering), "
        "classified per-position: **both_correct** (o1=d1 and o2=d2), "
        "**partial** (one position correct), **both_wrong**."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| failure_reason | both_correct | partial | both_wrong | total | % of all |")
    lines.append("|---|---|---|---|---|---|")

    for reason in ordered:
        g = merged[merged["failure_reason"] == reason]
        bc, pa, bw, tot = _correctness_row(g)
        pct = 100 * tot / total
        lines.append(f"| `{reason}` | {bc} | {pa} | {bw} | {tot} | {pct:.1f}% |")

    bc_tot, pa_tot, bw_tot, _ = _correctness_row(merged)
    lines.append(f"| **TOTAL** | **{bc_tot}** | **{pa_tot}** | **{bw_tot}** | **{total}** | 100% |")
    lines.append("")

    lines.append("## Per-Reason Breakdown")
    lines.append("")

    for reason in ordered:
        if reason in SUMMARY_ONLY:
            continue
        g = merged[merged["failure_reason"] == reason]
        bc, pa, bw, tot = _correctness_row(g)
        desc = FAILURE_DESCRIPTIONS.get(reason, "_No description available._")

        lines.append(f"### `{reason}` ({tot} samples)")
        lines.append("")
        lines.append(f"**Correctness:** {bc} both_correct / {pa} partial / {bw} both_wrong")
        lines.append("")
        lines.append(desc)
        lines.append("")

        if tot > 0:
            lines.append(f"**Examples** (up to {N_EXAMPLES} of {tot}):")
            lines.append("")
            lines.append(_examples_table(g))
            lines.append("")

        if visuals and reason in visuals:
            plot_rel, crossover_md = visuals[reason]
            if plot_rel:
                lines.append(
                    f"**Steering experiment** "
                    f"(scale range {STEERING_SCALE_RANGE[0]}-{STEERING_SCALE_RANGE[1]}, "
                    f"step {STEERING_STEP}):"
                )
                lines.append("")
                lines.append(f"![Steering plot for {reason}]({plot_rel})")
                lines.append("")
            if crossover_md:
                lines.append("**Crossover analysis:**")
                lines.append("")
                lines.append(crossover_md)
                lines.append("")

    return "\n".join(lines)
