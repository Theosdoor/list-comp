"""
Aggregate 2-layer sweep results from W&B into a LaTeX table and CSV cache.

Usage:
  python scripts/make_2layer_table.py \
    --flags-sweep-ids <id1> [<id2> ...] \
    --dmodel-sweep-ids <id1> [<id2> ...]

  # Offline reuse (skip W&B fetch):
  python scripts/make_2layer_table.py --use-cache

Output:
  - results/2layer_sweep_cache.csv  (one row per completed run)
  - LaTeX tables printed to stdout
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os

import pandas as pd
import wandb
from dotenv import load_dotenv


CACHE_PATH     = "results/2layer_sweep_cache.csv"
BOLD_THRESHOLD = 0.90
WANDB_ENTITY   = "theo-farrell99-durham-university"
WANDB_PROJECT  = "order-by-scale"

CONFIG_COLS = ["d_model", "use_ln", "use_bias", "use_wv", "use_wo", "use_mlp"]


# ---------------------------------------------------------------------------
# W&B fetch
# ---------------------------------------------------------------------------

def fetch_runs(sweep_ids):
    api  = wandb.Api()
    rows = []
    for sweep_id in sweep_ids:
        print(f"Fetching sweep: {sweep_id} ...")
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"sweep": sweep_id},
        )
        for run in runs:
            acc = run.summary.get("final/val_accuracy", None)
            if acc is None:
                continue  # skip in-progress / crashed runs
            c = run.config
            rows.append({
                "run_name":           run.name,
                "sweep_id":           sweep_id,
                "d_model":            int(c.get("d_model", 64)),
                "use_ln":             bool(c.get("use_ln",   False)),
                "use_bias":           bool(c.get("use_bias", False)),
                "use_wv":             bool(c.get("use_wv",   False)),
                "use_wo":             bool(c.get("use_wo",   False)),
                "use_mlp":            bool(c.get("use_mlp",  False)),
                "seed":               int(c.get("seed", -1)),
                "final_val_accuracy": float(acc),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def build_stats(df):
    grouped = df.groupby(CONFIG_COLS)["final_val_accuracy"]
    stats   = grouped.agg(
        mean="mean",
        max="max",
        min="min",
        median="median",
        n_seeds="count",
    ).reset_index()
    return stats.sort_values(CONFIG_COLS)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _cell(val, bold):
    s = f"{val:.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def latex_table_full(stats_df):
    """Full table: all stats columns."""
    lines = [
        r"\begin{tabular}{ccccccrrrrr}",
        r"\toprule",
        r"$d$ & LN & Bias & $W_V$ & $W_O$ & MLP"
        r" & mean & max & min & median & $n$ \\",
        r"\midrule",
    ]
    for _, r in stats_df.iterrows():
        good = r["mean"] >= BOLD_THRESHOLD
        cells = [
            str(int(r["d_model"])),
            "T" if r["use_ln"]   else "F",
            "T" if r["use_bias"] else "F",
            "T" if r["use_wv"]   else "F",
            "T" if r["use_wo"]   else "F",
            "T" if r["use_mlp"]  else "F",
            _cell(r["mean"],   good),
            _cell(r["max"],    good),
            _cell(r["min"],    good),
            _cell(r["median"], good),
            str(int(r["n_seeds"])),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_table_compact(stats_df, d_model=64):
    """Compact table (mean only) for flag sweep at a fixed d_model."""
    subset = stats_df[stats_df["d_model"] == d_model]
    lines = [
        r"\begin{tabular}{cccccr}",
        r"\toprule",
        r"LN & Bias & $W_V$ & $W_O$ & MLP & mean \\",
        r"\midrule",
    ]
    for _, r in subset.iterrows():
        good = r["mean"] >= BOLD_THRESHOLD
        cells = [
            "T" if r["use_ln"]   else "F",
            "T" if r["use_bias"] else "F",
            "T" if r["use_wv"]   else "F",
            "T" if r["use_wo"]   else "F",
            "T" if r["use_mlp"]  else "F",
            _cell(r["mean"], good),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flags-sweep-ids",  nargs="*", default=[], metavar="ID",
                        help="W&B sweep IDs for the flag grid (can pass multiple to merge)")
    parser.add_argument("--dmodel-sweep-ids", nargs="*", default=[], metavar="ID",
                        help="W&B sweep IDs for the d_model grid")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load from results/2layer_sweep_cache.csv instead of fetching W&B")
    args = parser.parse_args()

    load_dotenv()

    # ---- Load / fetch data ------------------------------------------------
    if args.use_cache and os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)
        for col in ["use_ln", "use_bias", "use_wv", "use_wo", "use_mlp"]:
            df[col] = df[col].astype(bool)
        print(f"Loaded {len(df)} runs from cache: {CACHE_PATH}")
    else:
        all_ids = args.flags_sweep_ids + args.dmodel_sweep_ids
        if not all_ids:
            parser.error("Provide --flags-sweep-ids and/or --dmodel-sweep-ids, or use --use-cache.")
        df = fetch_runs(all_ids)
        print(f"Fetched {len(df)} completed runs")
        if df.empty:
            print("No completed runs found.")
            return
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        print(f"Saved cache: {CACHE_PATH}")

    if df.empty:
        print("No data to process.")
        return

    stats = build_stats(df)

    # ---- Block 1: Flag sweep (d_model=64) ---------------------------------
    flags_stats = stats[stats["d_model"] == 64]
    if not flags_stats.empty:
        print("\n\n=== BLOCK 1: Flag Sweep (d_model=64) ===\n")
        print(latex_table_full(flags_stats))
        print("\n--- Compact (mean only) ---\n")
        print(latex_table_compact(flags_stats, d_model=64))

    # ---- Block 2: d_model scale (all-False flags) -------------------------
    all_false = ~(stats["use_ln"] | stats["use_bias"] | stats["use_wv"]
                  | stats["use_wo"] | stats["use_mlp"])
    dmodel_stats = stats[all_false]
    if not dmodel_stats.empty:
        print("\n\n=== BLOCK 2: d_model Scale (all FFFFF) ===\n")
        print(latex_table_full(dmodel_stats))

    # ---- Top-3 FFFFF d_model=64 runs for model selection ------------------
    mask_fffff64 = (
        (df["d_model"] == 64)
        & ~df["use_ln"] & ~df["use_bias"] & ~df["use_wv"]
        & ~df["use_wo"] & ~df["use_mlp"]
    )
    fffff64 = df[mask_fffff64]
    if not fffff64.empty:
        top3 = fffff64.nlargest(3, "final_val_accuracy")
        print("\n=== Top-3 FFFFF d_model=64 runs (keep in models/2_layer_sweep/) ===")
        for _, r in top3.iterrows():
            print(f"  {r['run_name']}  acc={r['final_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
