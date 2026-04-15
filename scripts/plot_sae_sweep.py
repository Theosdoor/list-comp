"""
SAE Sweep Summary: Table and Figure

Reads sae_comparison_*.md (most recent), aggregates metrics by (k, d_sae)
across seeds and learning rates, and produces:
  - A markdown + LaTeX summary table (EV, CE Increase, Dead %, N Special Features)
  - A scatter plot of CE Increase vs Explained Variance, coloured by k

Usage:
    python scripts/plot_sae_sweep.py
    python scripts/plot_sae_sweep.py --report path/to/sae_comparison.md
    python scripts/plot_sae_sweep.py --output-dir report/figures
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=None,
                   help="Path to sae_comparison_*.md (default: most recent in project root)")
    p.add_argument("--output-dir", type=Path, default=Path("report/figures"),
                   help="Directory to write figure and table files")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="k values to include (default: 1 2 3 4 5)")
    p.add_argument("--d-sae-values", type=int, nargs="+", default=None,
                   help="d_sae values to include (default: all)")
    return p.parse_args()

# ── Parsing ───────────────────────────────────────────────────────────────────

def find_latest_report(root: Path) -> Path:
    candidates = sorted(root.glob("sae_comparison_*.md"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No sae_comparison_*.md found in {root}")
    return candidates[0]


def parse_report(path: Path) -> pd.DataFrame:
    """Parse main summary table and special features table; join on model name."""
    text = path.read_text()

    # ── Main summary table ────────────────────────────────────────────────────
    # Columns: Model | d_sae | k | L0 | Dead | Dead% | Alive | MSE | Exp Var |
    #          Baseline Acc | Patched Acc | Acc Drop | Baseline CE | Patched CE | CE Increase
    summary_rows = []
    in_summary = False
    for line in text.splitlines():
        if "## Summary Table" in line:
            in_summary = True
            continue
        if in_summary and line.startswith("##"):
            break
        if in_summary and line.startswith("| sae_"):
            cols = [c.strip() for c in line.split("|")]
            # cols[0] = '', cols[1] = model, cols[2..] = values
            try:
                summary_rows.append({
                    "model":       cols[1],
                    "d_sae":       int(cols[2]),
                    "k":           int(cols[3]),
                    "ev":          float(cols[9]),
                    "dead_pct":    float(cols[6].rstrip("%")),
                    "baseline_ce": float(cols[13]),
                    "patched_ce":  float(cols[14]),
                    "ce_increase": float(cols[15]),
                })
            except (IndexError, ValueError):
                pass

    # ── Special features table ────────────────────────────────────────────────
    # Columns: Model | N Special | Special % | Max Corr | Mean Abs Corr
    special_rows = {}
    in_special = False
    for line in text.splitlines():
        if "## Special Features" in line:
            in_special = True
            continue
        if in_special and line.startswith("##"):
            break
        if in_special and line.startswith("| sae_"):
            cols = [c.strip() for c in line.split("|")]
            try:
                special_rows[cols[1]] = int(cols[2])
            except (IndexError, ValueError):
                pass

    df = pd.DataFrame(summary_rows)
    df["n_special"] = df["model"].map(special_rows).fillna(0).astype(int)
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame, k_values, d_sae_values) -> pd.DataFrame:
    """Group by (k, d_sae), compute mean ± std over seeds/lr runs."""
    if k_values:
        df = df[df["k"].isin(k_values)]
    if d_sae_values:
        df = df[df["d_sae"].isin(d_sae_values)]

    grouped = df.groupby(["k", "d_sae"])

    rows = []
    for (k, d_sae), g in grouped:
        n = len(g)
        rows.append({
            "k":              k,
            "d_sae":          d_sae,
            "n_runs":         n,
            "ev_mean":        g["ev"].mean(),
            "ev_std":         g["ev"].std(ddof=1) if n > 1 else 0.0,
            "ce_inc_mean":    g["ce_increase"].mean(),
            "ce_inc_std":     g["ce_increase"].std(ddof=1) if n > 1 else 0.0,
            "dead_pct_mean":  g["dead_pct"].mean(),
            "n_special_mean": g["n_special"].mean(),
            "n_special_std":  g["n_special"].std(ddof=1) if n > 1 else 0.0,
            # keep raw for scatter
            "ev_all":         g["ev"].values,
            "ce_inc_all":     g["ce_increase"].values,
        })

    return pd.DataFrame(rows).sort_values(["k", "d_sae"]).reset_index(drop=True)


# ── Table output ──────────────────────────────────────────────────────────────

def fmt(mean, std, decimals=4):
    """Format as 'mean ± std' with consistent decimal places."""
    if std == 0 or np.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def write_markdown_table(agg: pd.DataFrame, path: Path):
    lines = [
        "# SAE Sweep: Aggregated Metrics by (k, d_sae)\n",
        "Aggregated over all seeds and learning rates. "
        "Values shown as mean ± std.\n",
        "",
        "| k | d\\_sae | N runs | Exp Var | CE Increase | Dead % | N Special Feats |",
        "|---|--------|--------|---------|-------------|--------|----------------|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {int(r.k)} | {int(r.d_sae)} | {int(r.n_runs)} "
            f"| {fmt(r.ev_mean, r.ev_std)} "
            f"| {fmt(r.ce_inc_mean, r.ce_inc_std)} "
            f"| {r.dead_pct_mean:.1f}% "
            f"| {fmt(r.n_special_mean, r.n_special_std, decimals=2)} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"  Markdown table → {path}")


def write_latex_table(agg: pd.DataFrame, path: Path):
    """LaTeX booktabs table suitable for a dissertation."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SAE sweep results aggregated over seeds and learning rates "
        r"(mean $\pm$ std). Baseline CE = 0.1115 for all models.}",
        r"\label{tab:sae_sweep}",
        r"\begin{tabular}{cccrrrr}",
        r"\toprule",
        r"$k$ & $d_\text{SAE}$ & Runs & Exp.\ Var & CE Increase & Dead \% & $N_\text{special}$ \\",
        r"\midrule",
    ]
    prev_k = None
    for _, r in agg.iterrows():
        if prev_k is not None and int(r.k) != prev_k:
            lines.append(r"\midrule")
        prev_k = int(r.k)

        def lf(mean, std, dec=4):
            if std == 0 or np.isnan(std):
                return f"{mean:.{dec}f}"
            return f"${mean:.{dec}f} \\pm {std:.{dec}f}$"

        lines.append(
            f"  {int(r.k)} & {int(r.d_sae)} & {int(r.n_runs)} "
            f"& {lf(r.ev_mean, r.ev_std)} "
            f"& {lf(r.ce_inc_mean, r.ce_inc_std)} "
            f"& {r.dead_pct_mean:.1f} "
            f"& {lf(r.n_special_mean, r.n_special_std, dec=2)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  LaTeX table     → {path}")


# ── Figure ────────────────────────────────────────────────────────────────────

# Colour palette per k value — perceptually distinct, works in print
K_COLORS = {1: "#e41a1c", 2: "#ff7f00", 3: "#4daf4a", 4: "#377eb8", 5: "#984ea3"}
K_LABELS = {k: f"$k={k}$" for k in K_COLORS}


def plot_sweep(df_raw: pd.DataFrame, agg: pd.DataFrame, path: Path):
    """
    Two-panel figure:
      Left  — scatter of all runs (CE Increase vs EV), coloured by k
      Right — aggregated means with ± std error bars + d_sae annotations
    Both panels use a log y-axis (CE Increase spans ~2 orders of magnitude).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, use_agg in zip(axes, [False, True]):
        ax.set_xlabel("Explained Variance (↑ better)", fontsize=11)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.3f}" if x < 1 else f"{x:.1f}"
        ))
        ax.grid(True, alpha=0.3, linewidth=0.5, which="both")

        if not use_agg:
            for k, grp in df_raw.groupby("k"):
                color = K_COLORS.get(k, "grey")
                ax.scatter(grp["ev"], grp["ce_increase"],
                           color=color, alpha=0.3, s=14,
                           label=K_LABELS.get(k, str(k)))
            ax.set_ylabel("CE Increase (↓ better)", fontsize=11)
            ax.set_title("All runs", fontsize=11)
        else:
            for _, r in agg.iterrows():
                k = int(r.k)
                color = K_COLORS.get(k, "grey")
                label = K_LABELS.get(k) if r.d_sae == agg[agg.k == k]["d_sae"].min() else None
                ax.errorbar(r.ev_mean, r.ce_inc_mean,
                            xerr=r.ev_std, yerr=r.ce_inc_std,
                            fmt="o", color=color, markersize=7,
                            capsize=3, linewidth=1.2, label=label)
                # Annotate d_sae only for k=3 to avoid clutter
                if k == 3:
                    ax.annotate(f"d={int(r.d_sae)}",
                                xy=(r.ev_mean, r.ce_inc_mean),
                                xytext=(4, 2), textcoords="offset points",
                                fontsize=7, color=color)
            ax.set_title("Aggregated (mean ± std)", fontsize=11)

        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=9,
                  framealpha=0.9, loc="upper left")

    fig.suptitle("SAE Sweep: Reconstruction Fidelity vs Downstream CE Increase",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure          → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    root = Path(__file__).parent.parent
    report_path = args.report or find_latest_report(root)
    print(f"Reading: {report_path}")

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_report(report_path)
    print(f"Parsed {len(df)} SAE rows")

    agg = aggregate(df, args.k_values, args.d_sae_values)
    print(f"Aggregated to {len(agg)} (k, d_sae) groups")

    write_markdown_table(agg, output_dir / "sae_sweep_table.md")
    write_latex_table(agg, output_dir / "sae_sweep_table.tex")
    plot_sweep(df, agg, output_dir / "sae_sweep_figure.pdf")
    # Also save as PNG for quick preview
    plot_sweep(df, agg, output_dir / "sae_sweep_figure.png")


if __name__ == "__main__":
    main()
