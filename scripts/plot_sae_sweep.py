"""
SAE Sweep Summary: Table and Figure

Reads sae_comparison_*.md (most recent), aggregates metrics by (l0, d_sae)
across seeds and learning rates, and produces:
  - A markdown + LaTeX summary table (L0, Loss Recovered, Dead %, N Special Features)
  - Two aggregated scatter plots:
      x = L0    (hue = d_sae, plasma palette)  — y = Loss Recovered
      x = d_sae (hue = L0,    tab10 palette)   — y = Loss Recovered

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
# python3 scripts/plot_sae_sweep.py --exclude-runs-col --no-table-errors --exclude-d-sae 100 448 512

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=None,
                   help="Path to sae_comparison_*.md (default: most recent in project root)")
    p.add_argument("--output-dir", type=Path, default=Path("results/compare_sae/figures"),
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
    p.add_argument("--exclude-special-col", action="store_true",
                   help="Omit the N Special Features column from tables")
    return p.parse_args()

# ── Parsing ───────────────────────────────────────────────────────────────────

def find_latest_report(root: Path) -> Path:
    report_dir = root / "results" / "compare_sae"
    candidates = sorted(report_dir.glob("sae_comparison_*.md"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No sae_comparison_*.md found in {report_dir}")
    return candidates[0]


def parse_report(path: Path) -> pd.DataFrame:
    """Parse the summary table from an sae_comparison_*.md report."""
    text = path.read_text()

    # Summary table columns:
    # New: | Model | d_sae | L0 | Dead% | LR | Exp Var | H_orig | H* | H0 | N Special (mean_abs_corr) |
    #        [1]    [2]    [3]   [4]    [5]   [6]      [7]      [8]  [9]  [10]
    # Old: | Model | d_sae | L0 | Dead% | LR | Exp Var | N Special |
    #        [1]    [2]    [3]   [4]    [5]   [6]      [7]
    # N Special cell may be "3 (0.742)" — parse only the integer count.
    rows = []
    in_summary = False
    for line in text.splitlines():
        if "## Summary Table" in line:
            in_summary = True
            continue
        if in_summary and line.startswith("##"):
            break
        if in_summary and line.startswith("| ") and not line.startswith("|---") and "| Model |" not in line:
            cols = [c.strip() for c in line.split("|")]
            # Support both old (7-col) and new (10-col) table formats
            n_cols = len(cols) - 2  # subtract leading/trailing empty strings from split
            n_special_col = 10 if n_cols >= 10 else 7
            try:
                rows.append({
                    "model":          cols[1],
                    "d_sae":          int(cols[2]),
                    "l0":             int(round(float(cols[3]))),
                    "dead_pct":       float(cols[4].rstrip("%")),
                    "loss_recovered": float(cols[5]) if cols[5] != "—" else None,
                    # Cell may be "3 (0.742)" — take only the leading integer count
                    "n_special":      int(cols[n_special_col].split()[0]) if cols[n_special_col] not in ("—", "") else 0,
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
            "l0":                    l0,
            "d_sae":                 d_sae,
            "n_runs":                n,
            "loss_recovered_mean":   g["loss_recovered"].dropna().mean(),
            "loss_recovered_std":    g["loss_recovered"].dropna().std(ddof=1) if len(g["loss_recovered"].dropna()) > 1 else 0.0,
            "dead_pct_mean":         g["dead_pct"].mean(),
            "dead_pct_std":          g["dead_pct"].std(ddof=1) if n > 1 else 0.0,
            "n_special_mean":        g["n_special"].mean(),
            "n_special_std":         g["n_special"].std(ddof=1) if n > 1 else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["l0", "d_sae"]).reset_index(drop=True)


# ── Table output ──────────────────────────────────────────────────────────────

def fmt(mean, std, decimals=4, no_errors=False):
    """Format as 'mean ± std' (or just 'mean' when no_errors=True)."""
    if no_errors or std == 0 or np.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def write_markdown_table(agg: pd.DataFrame, path: Path,
                         no_errors: bool = False, exclude_runs_col: bool = False,
                         exclude_special_col: bool = False):
    runs_hdr = "" if exclude_runs_col else " N runs |"
    special_hdr = "" if exclude_special_col else " N Special Feats |"
    runs_sep = "" if exclude_runs_col else "--------|"
    special_sep = "" if exclude_special_col else "----------------|"
    lines = [
        "# SAE Sweep: Aggregated Metrics by (L0, d_sae)\n",
        "Aggregated over all seeds and learning rates. "
        + ("Values shown as means only.\n" if no_errors else "Values shown as mean ± std.\n"),
        "",
        f"| L0 | d\\_sae |{runs_hdr} Loss Recovered | Dead % |{special_hdr}",
        f"|----|--------|{runs_sep}----------------|--------|{special_sep}",
    ]
    for _, r in agg.iterrows():
        runs_cell = "" if exclude_runs_col else f" {int(r.n_runs)} |"
        special_cell = "" if exclude_special_col else f" {fmt(r.n_special_mean, r.n_special_std, decimals=2, no_errors=no_errors)} |"
        lines.append(
            f"| {int(r.l0)} | {int(r.d_sae)} |{runs_cell}"
            f" {fmt(r.loss_recovered_mean, r.loss_recovered_std, no_errors=no_errors)} "
            f"| {fmt(r.dead_pct_mean, r.dead_pct_std, decimals=1, no_errors=no_errors)}% "
            f"|{special_cell}"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"  Markdown table → {path}")


def write_latex_table(agg: pd.DataFrame, path: Path,
                      no_errors: bool = False, exclude_runs_col: bool = False,
                      exclude_special_col: bool = False):
    """LaTeX booktabs table suitable for a dissertation.

    Within each L0 section the best cell per column is bolded.
    The single best cell in the entire column is also underlined.
    Higher is better for Loss Recovered; lower is better for Patched CE and Dead %.
    """
    # ── pre-compute bests ─────────────────────────────────────────────────────
    # (col, higher_is_better)
    scored_cols = [
        ("loss_recovered_mean", True),
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

        Math-mode cells use \\mathbf{} on entire value (mean, ±, std).
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

        # Math-mode: bold the entire value including ± and std
        if no_errors or std == 0 or np.isnan(std):
            content = f"{mean:.{dec}f}"
        else:
            content = f"{mean:.{dec}f} \\pm {std:.{dec}f}"
        
        if bold:
            text = f"$\\mathbf{{{content}}}$"
        else:
            text = f"${content}$"
        return rf"\underline{{{text}}}" if is_global else text

    def lf(mean, std, dec=4):
        if no_errors or std == 0 or np.isnan(std):
            return f"{mean:.{dec}f}"
        return f"${mean:.{dec}f} \\pm {std:.{dec}f}$"

    # ── build table ───────────────────────────────────────────────────────────
    if exclude_special_col:
        ncols = "crr" if exclude_runs_col else "ccrr"
    else:
        ncols = "ccrr" if exclude_runs_col else "cccrr"
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SAE sweep results aggregated over seeds and learning rates (mean $\pm$ std). "
        r"$L_0$: mean active features per token. $d_\text{SAE}$: dictionary size. "
        r"\textbf{LR} (loss recovered): $(H^* - H_0) / (H_\text{orig} - H_0)$ where $H_\text{orig}$ is baseline CE, $H^*$ is SAE-patched CE, and $H_0$ is zero-ablation CE ($\uparrow$ better; 1 = perfect reconstruction). "
        r"\textbf{Dead\,\%}: percentage of dictionary features with zero activation across the evaluation set ($\downarrow$ better). " +
        (r"$N_\text{special}$: features whose activation strongly correlates with the SEP attention difference $\alpha_{d_1}-\alpha_{d_2}$, as discussed in Section \ref{s:res_rq2}. " if not exclude_special_col else "") +
        r"\textbf{Bold}: best within each $L_0$ block; \underline{\textbf{underlined}}: best overall.}",
        r"\label{tab:sae_sweep}",
        r"\begin{tabular}{" + ncols + r"}",
        r"\toprule",
    ]
    
    if exclude_special_col:
        header = (r"$L_0$ & $d_\text{SAE}$ & LR ($\uparrow$) & Dead \% ($\downarrow$) \\"
                  if exclude_runs_col else
                  r"$L_0$ & $d_\text{SAE}$ & Runs & LR ($\uparrow$) & Dead \% ($\downarrow$) \\")
    else:
        header = (r"$L_0$ & $d_\text{SAE}$ & LR ($\uparrow$) & Dead \% ($\downarrow$) & $N_\text{special}$ \\"
                  if exclude_runs_col else
                  r"$L_0$ & $d_\text{SAE}$ & Runs & LR ($\uparrow$) & Dead \% ($\downarrow$) & $N_\text{special}$ \\")
    
    lines.append(header)
    lines.append(r"\midrule")
    
    prev_l0 = None
    for _, r in agg.iterrows():
        l0 = int(r.l0)
        if prev_l0 is not None and l0 != prev_l0:
            lines.append(r"\midrule")
        prev_l0 = l0

        lr_cell   = _fmt(r.loss_recovered_mean, r.loss_recovered_std, "loss_recovered_mean", l0)
        dead_cell = _fmt(r.dead_pct_mean,  r.dead_pct_std,  "dead_pct_mean",   l0, dec=1, plain=True)

        runs_cell = "" if exclude_runs_col else f"& {int(r.n_runs)} "
        special_cell = "" if exclude_special_col else f"& {lf(r.n_special_mean, r.n_special_std, dec=2)} "
        lines.append(
            f"  {l0} & {int(r.d_sae)} {runs_cell}"
            f"& {lr_cell} & {dead_cell} "
            f"{special_cell}\\\\"
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
      Col 0  — y = Loss Recovered
      Col 1  — y = Patched CE Loss
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)

    l0_pal   = _qualitative_palette(df["l0"])
    dsae_pal = _sequential_palette(df["d_sae"], "plasma")

    # (x, y, hue, palette, xlabel, ylabel, ylim_top)
    panels = [
        ("l0",    "loss_recovered", "d_sae", dsae_pal, "← L0 (Lower is sparser)",    "↑ Loss Recovered", 1.0),
        ("d_sae", "loss_recovered", "l0",    l0_pal,   "d_sae",                       "↑ Loss Recovered", 1.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, (ax, (x, y, hue, palette, xlabel, ylabel, ylim_top)) in enumerate(zip(axes.flat, panels)):
        sns.pointplot(
            data=df, x=x, y=y, hue=hue, palette=palette,
            errorbar="sd", markers="o", linestyles="-",
            capsize=0.08,
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
    if df.empty:
        print(f"ERROR: No rows parsed from {report_path}. "
              "Check that the report contains a Summary Table with rows starting '| sae_' or '| btk_'.")
        return

    df = filter_df(df, args.l0_values, args.d_sae_values,
                   exclude_l0=args.exclude_l0, exclude_d_sae=args.exclude_d_sae)
    agg = aggregate(df)
    print(f"Aggregated to {len(agg)} (L0, d_sae) groups")

    write_markdown_table(agg, output_dir / "sae_sweep_table.md",
                         no_errors=args.no_table_errors, exclude_runs_col=args.exclude_runs_col,
                         exclude_special_col=args.exclude_special_col)
    write_latex_table(agg, output_dir / "sae_sweep_table.tex",
                      no_errors=args.no_table_errors, exclude_runs_col=args.exclude_runs_col,
                      exclude_special_col=args.exclude_special_col)
    plot_sweep(df, output_dir / "sae_sweep_figure.pdf")
    plot_sweep(df, output_dir / "sae_sweep_figure.png")


if __name__ == "__main__":
    main()
