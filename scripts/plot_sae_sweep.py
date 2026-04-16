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
    p.add_argument("--no-table-errors", action="store_true",
                   help="Omit ± std from table cells, showing means only")
    p.add_argument("--exclude-runs-col", action="store_true",
                   help="Omit the N runs column from tables")
    return p.parse_args()

# ── Parsing ───────────────────────────────────────────────────────────────────

def find_latest_report(root: Path) -> Path:
    candidates = sorted(root.glob("sae_comparison_*.md"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No sae_comparison_*.md found in {root}")
    return candidates[0]


def parse_report(path: Path) -> pd.DataFrame:
    """Parse the summary table from an sae_comparison_*.md report."""
    text = path.read_text()

    # ── Summary table columns ─────────────────────────────────────────────────
    # | Model | d_sae | k | L0 | Dead % | Exp Var | Baseline CE | Patched CE | CE Increase | N Special |
    #   [1]    [2]    [3]  [4]   [5]      [6]       [7]           [8]          [9]            [10]
    rows = []
    in_summary = False
    for line in text.splitlines():
        if "## Summary Table" in line:
            in_summary = True
            continue
        if in_summary and line.startswith("##"):
            break
        if in_summary and line.startswith("| sae_"):
            cols = [c.strip() for c in line.split("|")]
            try:
                rows.append({
                    "model":       cols[1],
                    "d_sae":       int(cols[2]),
                    "k":           int(cols[3]),
                    "l0":          int(round(float(cols[4]))),
                    "dead_pct":    float(cols[5].rstrip("%")),
                    "ev":          float(cols[6]),
                    "baseline_ce": float(cols[7]),
                    "patched_ce":  float(cols[8]),
                    "ce_increase": float(cols[9]),
                    "n_special":   int(cols[10]) if cols[10] != "—" else 0,
                })
            except (IndexError, ValueError):
                pass

    return pd.DataFrame(rows)


# ── Filtering & Aggregation ───────────────────────────────────────────────────

def filter_df(df: pd.DataFrame, l0_values, d_sae_values,
              exclude_l0=None, exclude_d_sae=None) -> pd.DataFrame:
    if l0_values:
        df = df[df["l0"].isin(l0_values)]
    if d_sae_values:
        df = df[df["d_sae"].isin(d_sae_values)]
    if exclude_l0:
        df = df[~df["l0"].isin(exclude_l0)]
    if exclude_d_sae:
        df = df[~df["d_sae"].isin(exclude_d_sae)]
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (l0, d_sae), compute mean ± std over seeds/lr runs."""
    rows = []
    for (l0, d_sae), g in df.groupby(["l0", "d_sae"]):
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
            "dead_pct_std":     g["dead_pct"].std(ddof=1) if n > 1 else 0.0,
            "n_special_mean":   g["n_special"].mean(),
            "n_special_std":    g["n_special"].std(ddof=1) if n > 1 else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["l0", "d_sae"]).reset_index(drop=True)


# ── Table output ──────────────────────────────────────────────────────────────

def fmt(mean, std, decimals=4, no_errors=False):
    """Format as 'mean ± std' (or just 'mean' when no_errors=True)."""
    if no_errors or std == 0 or np.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def write_markdown_table(agg: pd.DataFrame, path: Path,
                         no_errors: bool = False, exclude_runs_col: bool = False):
    runs_hdr = "" if exclude_runs_col else " N runs |"
    lines = [
        "# SAE Sweep: Aggregated Metrics by (L0, d_sae)\n",
        "Aggregated over all seeds and learning rates. "
        + ("Values shown as means only.\n" if no_errors else "Values shown as mean ± std.\n"),
        "",
        f"| L0 | d\\_sae |{runs_hdr} EV | PCE | Dead % | N Special Feats |",
        f"|----|--------|{'--------|' if not exclude_runs_col else ''}----|-----|--------|----------------|",
    ]
    for _, r in agg.iterrows():
        runs_cell = "" if exclude_runs_col else f" {int(r.n_runs)} |"
        lines.append(
            f"| {int(r.l0)} | {int(r.d_sae)} |{runs_cell}"
            f" {fmt(r.ev_mean, r.ev_std, no_errors=no_errors)} "
            f"| {fmt(r.patched_ce_mean, r.patched_ce_std, no_errors=no_errors)} "
            f"| {fmt(r.dead_pct_mean, r.dead_pct_std, decimals=1, no_errors=no_errors)}% "
            f"| {fmt(r.n_special_mean, r.n_special_std, decimals=2, no_errors=no_errors)} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"  Markdown table → {path}")


def write_latex_table(agg: pd.DataFrame, path: Path,
                      no_errors: bool = False, exclude_runs_col: bool = False):
    """LaTeX booktabs table suitable for a dissertation.

    Within each L0 section the best cell per column is bolded.
    The single best cell in the entire column is also underlined.
    Higher is better for Exp Var; lower is better for Patched CE and Dead %.
    """
    # ── pre-compute bests ─────────────────────────────────────────────────────
    # (col, higher_is_better)
    scored_cols = [
        ("ev_mean",         True),
        ("patched_ce_mean", False),
        ("dead_pct_mean",   False),
    ]
    global_best = {
        col: (agg[col].max() if hi else agg[col].min())
        for col, hi in scored_cols
    }
    section_best = {
        l0: {
            col: (grp[col].max() if hi else grp[col].min())
            for col, hi in scored_cols
        }
        for l0, grp in agg.groupby("l0")
    }

    def _fmt(mean, std, col, l0, dec=4, plain=False) -> str:
        """Format a cell with bold (section best) or underline+bold (global best).

        Math-mode cells use \\mathbf{} on the mean so bold renders inside $.
        Plain-text cells (dead %) use \\textbf{}.
        """
        is_global  = np.isclose(mean, global_best[col], rtol=1e-6)
        is_section = np.isclose(mean, section_best[l0][col], rtol=1e-6)
        bold = is_global or is_section

        if plain:
            mean_str = f"{mean:.{dec}f}"
            if not no_errors and std > 0 and not np.isnan(std):
                mean_str = rf"{mean_str} $\pm$ {std:.{dec}f}"
            mean_str = rf"\textbf{{{mean_str}}}" if bold else mean_str
            return rf"\underline{{{mean_str}}}" if is_global else mean_str

        # Math-mode: bold only the mean, leave ±std unbolded
        mean_str = (rf"\mathbf{{{mean:.{dec}f}}}" if bold else f"{mean:.{dec}f}")
        if no_errors or std == 0 or np.isnan(std):
            text = f"${mean_str}$"
        else:
            text = f"${mean_str} \\pm {std:.{dec}f}$"
        return rf"\underline{{{text}}}" if is_global else text

    def lf(mean, std, dec=4):
        if no_errors or std == 0 or np.isnan(std):
            return f"{mean:.{dec}f}"
        return f"${mean:.{dec}f} \\pm {std:.{dec}f}$"

    # ── build table ───────────────────────────────────────────────────────────
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SAE sweep results aggregated over seeds and learning rates (mean $\pm$ std). "
        r"$L_0$: mean active features per token. $d_\text{SAE}$: dictionary size. "
        r"\textbf{EV} (explained variance): fraction of activation variance recovered by the SAE reconstruction ($\uparrow$ better). "
        r"\textbf{PCE} (patched cross-entropy): language-model CE after substituting SAE reconstructions for true activations ($\downarrow$ better; baseline $= 0.1115$). "
        r"\textbf{Dead\,\%}: percentage of dictionary features with zero activation across the evaluation set ($\downarrow$ better). "
        r"$N_\text{special}$: features whose activation strongly correlates with the SEP attention difference $\alpha_{d_1}-\alpha_{d_2}$, as discussed in Section \ref{s:res_rq2}. "
        r"\textbf{Bold}: best within each $L_0$ block; \underline{\textbf{underlined}}: best overall.}",
        r"\label{tab:sae_sweep}",
        r"\begin{tabular}{" + ("ccrrrrr" if exclude_runs_col else "cccrrrrr") + r"}",
        r"\toprule",
        (r"$L_0$ & $d_\text{SAE}$ & EV ($\uparrow$) & PCE ($\downarrow$) & Dead \% ($\downarrow$) & $N_\text{special}$ \\"
         if exclude_runs_col else
         r"$L_0$ & $d_\text{SAE}$ & Runs & EV ($\uparrow$) & PCE ($\downarrow$) & Dead \% ($\downarrow$) & $N_\text{special}$ \\"),
        r"\midrule",
    ]
    prev_l0 = None
    for _, r in agg.iterrows():
        l0 = int(r.l0)
        if prev_l0 is not None and l0 != prev_l0:
            lines.append(r"\midrule")
        prev_l0 = l0

        ev_cell   = _fmt(r.ev_mean,        r.ev_std,        "ev_mean",         l0)
        ce_cell   = _fmt(r.patched_ce_mean, r.patched_ce_std, "patched_ce_mean", l0)
        dead_cell = _fmt(r.dead_pct_mean,  r.dead_pct_std,  "dead_pct_mean",   l0, dec=1, plain=True)

        runs_cell = "" if exclude_runs_col else f"& {int(r.n_runs)} "
        lines.append(
            f"  {l0} & {int(r.d_sae)} {runs_cell}"
            f"& {ev_cell} & {ce_cell} & {dead_cell} "
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

def _qualitative_palette(values) -> dict:
    vals = sorted(set(values))
    cmap = plt.get_cmap("tab10")
    return {v: cmap(i % 10) for i, v in enumerate(vals)}


def _sequential_palette(values, cmap_name: str) -> dict:
    vals = sorted(set(values))
    cmap = plt.get_cmap(cmap_name)
    if len(vals) == 1:
        return {vals[0]: cmap(0.65)}
    return {v: cmap(0.15 + 0.7 * i / (len(vals) - 1)) for i, v in enumerate(vals)}


def plot_sweep(df: pd.DataFrame, path: Path):
    """
    2×2 grid of seaborn pointplots (±1 std over seeds/lr runs):
      Row 0  — x = L0    (hue = d_sae, plasma sequential)
      Row 1  — x = d_sae (hue = L0,    tab10 qualitative)
      Col 0  — y = % Variance Explained
      Col 1  — y = Patched CE Loss
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)

    plot_df = df.assign(ev_pct=df["ev"] * 100)

    l0_pal   = _qualitative_palette(df["l0"])
    dsae_pal = _sequential_palette(df["d_sae"], "plasma")

    # (x, y, hue, palette, xlabel, ylabel, ylim_top)
    panels = [
        ("l0",    "ev_pct",     "d_sae", dsae_pal, "← L0 (Lower is sparser)",    "↑ % Variance Explained", 100),
        ("l0",    "patched_ce", "d_sae", dsae_pal, "← L0 (Lower is sparser)",    "← Patched CE Loss",       None),
        ("d_sae", "ev_pct",     "l0",    l0_pal,   "d_sae", "↑ % Variance Explained", 100),
        ("d_sae", "patched_ce", "l0",    l0_pal,   "d_sae", "← Patched CE Loss",       None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for i, (ax, (x, y, hue, palette, xlabel, ylabel, ylim_top)) in enumerate(zip(axes.flat, panels)):
        sns.pointplot(
            data=plot_df, x=x, y=y, hue=hue, palette=palette,
            errorbar="sd", markers="o", linestyles="-",
            err_kws={"linewidth": 1.5}, capsize=0.08,
            ax=ax
        )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(bottom=0, top=ylim_top)
        if i % 2 == 0:  # left panels
            ax.get_legend().remove()
        else:
            ax.legend(fontsize=8, framealpha=0.9, loc="upper right",
                      ncol=2 if hue == "d_sae" else 1)

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

    df = filter_df(df, args.l0_values, args.d_sae_values,
                   exclude_l0=args.exclude_l0, exclude_d_sae=args.exclude_d_sae)
    agg = aggregate(df)
    print(f"Aggregated to {len(agg)} (L0, d_sae) groups")

    write_markdown_table(agg, output_dir / "sae_sweep_table.md",
                         no_errors=args.no_table_errors, exclude_runs_col=args.exclude_runs_col)
    write_latex_table(agg, output_dir / "sae_sweep_table.tex",
                      no_errors=args.no_table_errors, exclude_runs_col=args.exclude_runs_col)
    plot_sweep(df, output_dir / "sae_sweep_figure.pdf")
    plot_sweep(df, output_dir / "sae_sweep_figure.png")


if __name__ == "__main__":
    main()
