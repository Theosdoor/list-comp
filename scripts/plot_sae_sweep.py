"""
SAE Sweep Summary: Table and Figure

Reads sae_comparison_*.md (most recent), aggregates metrics by (l0, d_sae)
across seeds and learning rates, and produces:
  - A markdown + LaTeX summary table (L0, EV, Patched CE, Dead %, N Special Features)
  - Two aggregated scatter plots of Patched CE vs % Variance Explained:
      left  — hue by L0   (viridis palette)
      right — hue by d_sae (plasma palette)

Usage:
    python scripts/plot_sae_sweep.py
    python scripts/plot_sae_sweep.py --report path/to/sae_comparison.md
    python scripts/plot_sae_sweep.py --output-dir report/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=None,
                   help="Path to sae_comparison_*.md (default: most recent in project root)")
    p.add_argument("--output-dir", type=Path, default=Path("report/figures"),
                   help="Directory to write figure and table files")
    p.add_argument("--l0-values", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="L0 values to include (default: 1 2 3 4 5)")
    p.add_argument("--d-sae-values", type=int, nargs="+", default=None,
                   help="d_sae values to include (default: all)")
    p.add_argument("--exclude-l0", type=int, nargs="+", default=None,
                   help="L0 values to exclude (e.g. --exclude-l0 1)")
    p.add_argument("--exclude-d-sae", type=int, nargs="+", default=None,
                   help="d_sae values to exclude (e.g. --exclude-d-sae 64)")
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
                    "l0":          int(round(float(cols[4]))),
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

def aggregate(df: pd.DataFrame, l0_values, d_sae_values,
              exclude_l0=None, exclude_d_sae=None) -> pd.DataFrame:
    """Group by (l0, d_sae), compute mean ± std over seeds/lr runs."""
    if l0_values:
        df = df[df["l0"].isin(l0_values)]
    if d_sae_values:
        df = df[df["d_sae"].isin(d_sae_values)]
    if exclude_l0:
        df = df[~df["l0"].isin(exclude_l0)]
    if exclude_d_sae:
        df = df[~df["d_sae"].isin(exclude_d_sae)]

    grouped = df.groupby(["l0", "d_sae"])

    rows = []
    for (l0, d_sae), g in grouped:
        n = len(g)
        rows.append({
            "l0":               l0,
            "d_sae":            d_sae,
            "n_runs":           n,
            "ev_mean":          g["ev"].mean(),
            "ev_std":           g["ev"].std(ddof=1) if n > 1 else 0.0,
            "patched_ce_mean":  g["patched_ce"].mean(),
            "patched_ce_std":   g["patched_ce"].std(ddof=1) if n > 1 else 0.0,
            "dead_pct_mean":    g["dead_pct"].mean(),
            "n_special_mean":   g["n_special"].mean(),
            "n_special_std":    g["n_special"].std(ddof=1) if n > 1 else 0.0,
        })

    return pd.DataFrame(rows).sort_values(["l0", "d_sae"]).reset_index(drop=True)


# ── Table output ──────────────────────────────────────────────────────────────

def fmt(mean, std, decimals=4):
    """Format as 'mean ± std' with consistent decimal places."""
    if std == 0 or np.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def write_markdown_table(agg: pd.DataFrame, path: Path):
    lines = [
        "# SAE Sweep: Aggregated Metrics by (L0, d_sae)\n",
        "Aggregated over all seeds and learning rates. "
        "Values shown as mean ± std.\n",
        "",
        "| L0 | d\\_sae | N runs | Exp Var | Patched CE | Dead % | N Special Feats |",
        "|----|--------|--------|---------|------------|--------|----------------|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {int(r.l0)} | {int(r.d_sae)} | {int(r.n_runs)} "
            f"| {fmt(r.ev_mean, r.ev_std)} "
            f"| {fmt(r.patched_ce_mean, r.patched_ce_std)} "
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
        r"\begin{tabular}{cccrrrrr}",
        r"\toprule",
        r"$L_0$ & $d_\text{SAE}$ & Runs & Exp.\ Var & Patched CE & Dead \% & $N_\text{special}$ \\",
        r"\midrule",
    ]
    prev_l0 = None
    for _, r in agg.iterrows():
        if prev_l0 is not None and int(r.l0) != prev_l0:
            lines.append(r"\midrule")
        prev_l0 = int(r.l0)

        def lf(mean, std, dec=4):
            if std == 0 or np.isnan(std):
                return f"{mean:.{dec}f}"
            return f"${mean:.{dec}f} \\pm {std:.{dec}f}$"

        lines.append(
            f"  {int(r.l0)} & {int(r.d_sae)} & {int(r.n_runs)} "
            f"& {lf(r.ev_mean, r.ev_std)} "
            f"& {lf(r.patched_ce_mean, r.patched_ce_std)} "
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

def _qualitative_colors(values) -> dict:
    """Map sorted unique values to maximally-distinct tab10 hues (categorical)."""
    vals = sorted(set(values))
    cmap = plt.get_cmap("tab10")
    return {v: cmap(i % 10) for i, v in enumerate(vals)}


def _sequential_colors(values, cmap_name: str) -> dict:
    """Map sorted unique values to colours from a sequential colormap."""
    vals = sorted(set(values))
    cmap = plt.get_cmap(cmap_name)
    if len(vals) == 1:
        return {vals[0]: cmap(0.65)}
    return {v: cmap(0.15 + 0.7 * i / (len(vals) - 1)) for i, v in enumerate(vals)}


def _legend_handles(color_map: dict, label_fmt: str) -> list:
    """Build Line2D legend handles from a {value: color} mapping."""
    return [
        Line2D([0], [0], marker="o", color=c, linestyle="None",
               markersize=7, label=label_fmt.format(v))
        for v, c in sorted(color_map.items())
    ]


def _clipped_errorbar(mean, std, lo=None, hi=None):
    """
    Return asymmetric (lower, upper) error bar magnitudes, clipped so that
    mean - lower >= lo and mean + upper <= hi. Returns None if std == 0.
    """
    if std == 0 or np.isnan(std):
        return None
    lower = std if lo is None else min(std, mean - lo)
    upper = std if hi is None else min(std, hi - mean)
    return [[max(lower, 0)], [max(upper, 0)]]


def plot_sweep(agg: pd.DataFrame, path: Path):
    """
    Two-panel figure, both showing Patched CE Loss vs % Variance Explained,
    aggregated (mean ± std over seeds / lr runs):
      Left  — hue by L0    (tab10 qualitative palette)
      Right — hue by d_sae (plasma sequential palette)
    Error bars are clipped to [0, 100] on x and [0, ∞) on y.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)

    l0_colors   = _qualitative_colors(agg["l0"])
    dsae_colors = _sequential_colors(agg["d_sae"], "plasma")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, hue_col, color_map, label_fmt in [
        (axes[0], "l0",    l0_colors,   "L0 = {}"),
        (axes[1], "d_sae", dsae_colors, "$d_{{\\rm SAE}}$ = {}"),
    ]:
        for hue_val, grp in agg.groupby(hue_col):
            color = color_map[hue_val]
            for _, r in grp.iterrows():
                x = r.ev_mean * 100
                xerr = _clipped_errorbar(x, r.ev_std * 100, lo=0.0, hi=100.0)
                yerr = _clipped_errorbar(r.patched_ce_mean, r.patched_ce_std, lo=0.0)
                ax.errorbar(
                    x, r.patched_ce_mean,
                    xerr=xerr, yerr=yerr,
                    fmt="o", color=color, markersize=7,
                    capsize=3, linewidth=1.2,
                )

        ax.set_xlabel("% Variance Explained (↑ better)", fontsize=11)
        ax.set_ylabel("Patched CE Loss (↓ better)", fontsize=11)
        ax.set_xlim(right=100.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(
            handles=_legend_handles(color_map, label_fmt),
            fontsize=9, framealpha=0.9, loc="upper right",
        )

    fig.suptitle("SAE Sweep: Reconstruction Fidelity vs Patched CE Loss",
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

    agg = aggregate(df, args.l0_values, args.d_sae_values,
                    exclude_l0=args.exclude_l0, exclude_d_sae=args.exclude_d_sae)
    print(f"Aggregated to {len(agg)} (L0, d_sae) groups")

    write_markdown_table(agg, output_dir / "sae_sweep_table.md")
    write_latex_table(agg, output_dir / "sae_sweep_table.tex")
    plot_sweep(agg, output_dir / "sae_sweep_figure.pdf")
    plot_sweep(agg, output_dir / "sae_sweep_figure.png")


if __name__ == "__main__":
    main()
