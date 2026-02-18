"""
Analyze swap analysis failure reasons for SAE feature crossover results.

Loads xovers and swap_bounds CSVs, classifies base-model correctness at scale=1.0,
then generates a markdown report breaking down each failure reason with counts,
correctness distribution, and concrete examples.

Usage:
    python3 scripts/analyze_failure_reasons.py [--feature 30] [--output results/xover/failure_analysis_feat30.md]
"""
import sys
import ast
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/xover")
SCALE_START = 0.0
SCALE_STEP = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", type=int, default=30)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def parse_list_field(field):
    if isinstance(field, str):
        return ast.literal_eval(field)
    if isinstance(field, (list, np.ndarray)):
        return list(field)
    return field


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(feature_idx):
    xovers_path = RESULTS_DIR / f"xovers_feat{feature_idx}.csv"
    bounds_path = RESULTS_DIR / f"swap_bounds_feat{feature_idx}.csv"
    xovers_df = pd.read_csv(xovers_path)
    bounds_df = pd.read_csv(bounds_path)
    return xovers_df, bounds_df


def get_correctness(xovers_df):
    """
    Determine base model correctness at scale=1.0 for each sample.
    Uses argmax_o1/argmax_o2 at the scale-1.0 grid index from xovers_df.
    """
    scale_1_idx = int(round((1.0 - SCALE_START) / SCALE_STEP))
    rows = []
    for _, row in xovers_df.iterrows():
        argmax_o1 = parse_list_field(row["argmax_o1"])
        argmax_o2 = parse_list_field(row["argmax_o2"])
        d1, d2 = int(row["d1"]), int(row["d2"])
        pred_o1 = argmax_o1[scale_1_idx]
        pred_o2 = argmax_o2[scale_1_idx]
        o1c = pred_o1 == d1
        o2c = pred_o2 == d2
        if o1c and o2c:
            correctness = "both_correct"
        elif o1c or o2c:
            correctness = "partial"
        else:
            correctness = "both_wrong"
        rows.append({"d1": d1, "d2": d2, "correctness": correctness})
    return pd.DataFrame(rows)


def build_merged(xovers_df, bounds_df):
    """Merge swap bounds (failure reasons) with xovers data and correctness labels."""
    correctness_df = get_correctness(xovers_df)

    # Normalise failure_reason: NaN -> 'success'
    bounds_df = bounds_df.copy()
    bounds_df["failure_reason"] = bounds_df["failure_reason"].fillna("success")

    merged = bounds_df.merge(correctness_df, on=["d1", "d2"], how="left")
    # Attach selected xovers columns for examples
    xovers_slim = xovers_df[
        ["d1", "d2", "feat_orig", "o1_crossovers", "o2_crossovers",
         "o1_bound_types", "o2_bound_types", "n_o1_xover", "n_o2_xover",
         "o1_failure_reason"]
    ].copy()
    merged = merged.merge(xovers_slim, on=["d1", "d2"], how="left")
    return merged


# ── Markdown generation ───────────────────────────────────────────────────────

FAILURE_DESCRIPTIONS = {
    "success": (
        "The pipeline found a valid swap zone: both o1 and o2 crossovers resolved correctly "
        "and argmax dominance confirmed a contiguous scale window where o1 predicts d2 and o2 "
        "predicts d1."
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
        "The analytical o1 crossover is beyond scale 20 (2× the grid ceiling). "
        "The linear model is extrapolating far outside tested territory, so we flag "
        "rather than trust the value."
    ),
    "no_o2_crossover": (
        "No sign change in the d1−d2 logit diff at o2 across the whole scale grid. "
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


def correctness_row(group):
    counts = group["correctness"].value_counts()
    bc = counts.get("both_correct", 0)
    pa = counts.get("partial", 0)
    bw = counts.get("both_wrong", 0)
    tot = len(group)
    return bc, pa, bw, tot


def fmt_example(row, n=5):
    """Format a single example row as a compact markdown table row."""
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


def examples_table(group, n=5):
    sample = group.head(n)
    lines = [
        "| d1 | d2 | feat_orig | o1_crossovers | o2_crossovers | lower_bound | upper_bound | correctness |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in sample.iterrows():
        lines.append(fmt_example(row))
    return "\n".join(lines)


def generate_markdown(merged, feature_idx):
    total = len(merged)
    lines = []

    lines.append(f"# Failure Reason Analysis — Feature {feature_idx}")
    lines.append("")
    lines.append(
        "Pipeline: `get_xovers_df` → `get_output_swap_bounds`. "
        "Correctness is the base model accuracy at scale=1.0 (unsteered), "
        "classified per-position: **both_correct** (o1=d1 and o2=d2), "
        "**partial** (one position correct), **both_wrong**."
    )
    lines.append("")

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    lines.append("| failure_reason | both_correct | partial | both_wrong | total | % of all |")
    lines.append("|---|---|---|---|---|---|")

    present_reasons = merged["failure_reason"].value_counts().index.tolist()
    ordered = [r for r in FAILURE_ORDER if r in present_reasons]
    ordered += [r for r in present_reasons if r not in FAILURE_ORDER]

    for reason in ordered:
        g = merged[merged["failure_reason"] == reason]
        bc, pa, bw, tot = correctness_row(g)
        pct = 100 * tot / total
        lines.append(f"| `{reason}` | {bc} | {pa} | {bw} | {tot} | {pct:.1f}% |")

    bc_tot, pa_tot, bw_tot, _ = correctness_row(merged)
    lines.append(f"| **TOTAL** | **{bc_tot}** | **{pa_tot}** | **{bw_tot}** | **{total}** | 100% |")
    lines.append("")

    # ── Per-reason sections ───────────────────────────────────────────────────
    lines.append("## Per-Reason Breakdown")
    lines.append("")

    for reason in ordered:
        g = merged[merged["failure_reason"] == reason]
        bc, pa, bw, tot = correctness_row(g)
        desc = FAILURE_DESCRIPTIONS.get(reason, "_No description available._")

        lines.append(f"### `{reason}` ({tot} samples)")
        lines.append("")
        lines.append(f"**Correctness:** {bc} both_correct / {pa} partial / {bw} both_wrong")
        lines.append("")
        lines.append(desc)
        lines.append("")

        if tot > 0:
            lines.append(f"**Examples** (up to 5 of {tot}):")
            lines.append("")
            lines.append(examples_table(g))
            lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    feature_idx = args.feature
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"failure_analysis_feat{feature_idx}.md"

    print(f"Loading data for feature {feature_idx}...")
    xovers_df, bounds_df = load_data(feature_idx)

    print("Building merged dataset with correctness labels...")
    merged = build_merged(xovers_df, bounds_df)

    print("Generating markdown...")
    md = generate_markdown(merged, feature_idx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
