"""
Special Latent vs SAE Performance Analysis

Shows that higher-performing SAEs reliably learn more special latents.
Produces two seaborn swarm plots:
  1) n_special_features (binned: 0, 1, 2, >2) vs patched CE increase
  2) n_special_features (binned: 0, 1, 2, >2) vs explained variance

Also prints Spearman correlations between n_special_features and each metric.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
import wandb
import argparse

# ── Config ─────────────────────────────────────────────────────────────────────

SWEEP    = "theo-farrell99-durham-university/orderbyscale_sae_sweep"
SAVE_DIR = Path(__file__).parent.parent / "results" / "special_latent_analysis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Parse Command-Line Arguments ───────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Special Latent vs SAE Performance Analysis - generates all visualization variants",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--ci-method",
    choices=["bootstrap", "sem", "both"],
    default="bootstrap",
    help="Method for computing confidence intervals in summary table (default: bootstrap)"
)

args = parser.parse_args()

# ── Statistical Helper Functions ───────────────────────────────────────────────

def bootstrap_ci(data, ci=0.95, n_bootstrap=10000):
    """
    Compute confidence interval via bootstrapping.
    
    Args:
        data: 1D array of values
        ci: confidence level (default 0.95 for 95% CI)
        n_bootstrap: number of bootstrap samples
    
    Returns:
        (lower, upper) bounds of confidence interval
    """
    data = np.asarray(data)
    n = len(data)
    bootstrap_means = []
    
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper


def sem_ci(data, ci=0.95):
    """
    Compute confidence interval via standard error of mean.
    
    Args:
        data: 1D array of values
        ci: confidence level (default 0.95 for 95% CI)
    
    Returns:
        (lower, upper) bounds of confidence interval
    """
    data = np.asarray(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    z = 1.96  # Approximate z-value for 95% CI
    lower = mean - z * sem
    upper = mean + z * sem
    
    return lower, upper


def compute_cis_for_category(df_cat, metric_col, ci_method="bootstrap"):
    """
    Compute both bootstrap and SEM CIs for a metric in a category.
    
    Args:
        df_cat: DataFrame for a single category
        metric_col: column name (e.g., "ce_increase" or "explained_var")
        ci_method: "bootstrap", "sem", or "both"
    
    Returns:
        dict with keys like "bootstrap_lower", "bootstrap_upper", "sem_lower", "sem_upper"
    """
    data = df_cat[metric_col].values
    result = {}
    
    if ci_method in ["bootstrap", "both"]:
        bootstrap_lower, bootstrap_upper = bootstrap_ci(data)
        result["bootstrap_lower"] = bootstrap_lower
        result["bootstrap_upper"] = bootstrap_upper
    
    if ci_method in ["sem", "both"]:
        sem_lower, sem_upper = sem_ci(data)
        result["sem_lower"] = sem_lower
        result["sem_upper"] = sem_upper
    
    return result


def pairwise_comparisons(df, category_col, metric_col):
    """
    Run Mann-Whitney U tests comparing category 0 against all others.
    
    Args:
        df: DataFrame with data
        category_col: column name for categories (e.g., "n_special_bin")
        metric_col: column name for metric (e.g., "ce_increase")
    
    Returns:
        dict with category names as keys, (statistic, p_value) as values
    """
    cat_0_data = df[df[category_col] == "0"][metric_col].values
    results = {}
    
    for cat in ["1", "2", ">2"]:
        cat_data = df[df[category_col] == cat][metric_col].values
        if len(cat_data) > 0:
            stat, pval = mannwhitneyu(cat_0_data, cat_data, alternative="two-sided")
            results[f"0_vs_{cat}"] = (stat, pval)
    
    return results


def format_pvalue_label(pval):
    """Convert p-value to significance label."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return "ns"


def spearman_ci(r, n, ci=0.95):
    """
    Compute Spearman correlation confidence interval via Fisher's z-transformation.
    
    Args:
        r: Spearman correlation coefficient
        n: number of samples
        ci: confidence level (default 0.95 for 95% CI)
    
    Returns:
        (lower, upper) bounds of confidence interval
    """
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.96  # Approximate z-value for 95% CI
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def add_significance_brackets(ax, comparisons, pval_results, gap_frac=0.06, tick_frac=0.02):
    """
    Draw stacked bracket-style significance annotations on a categorical axis.

    Each bracket sits above the previous one, and the y-axis is expanded to
    accommodate all of them.

    Args:
        ax:           matplotlib Axes object
        comparisons:  list of (x_left_idx, x_right_idx, comp_key) tuples
        pval_results: dict mapping comp_key -> (statistic, p_value)
        gap_frac:     fraction of current y-range between successive brackets
        tick_frac:    fraction of current y-range for the vertical tick drops
    """
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    level = 0
    for x_left, x_right, comp_key in comparisons:
        if comp_key not in pval_results:
            continue

        pval = pval_results[comp_key][1]
        label = format_pvalue_label(pval)

        bracket_y  = y_max + y_range * gap_frac * (level + 1)
        tick_drop  = y_range * tick_frac
        label_pad  = y_range * 0.01

        # horizontal bar + vertical ticks
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [bracket_y - tick_drop, bracket_y, bracket_y, bracket_y - tick_drop],
            color="black", linewidth=1.2, clip_on=False,
        )
        ax.text(
            (x_left + x_right) / 2.0,
            bracket_y + label_pad,
            label,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            clip_on=False,
        )
        level += 1

    # Expand the y-axis so all brackets are visible
    if level > 0:
        ax.set_ylim(y_min, y_max + y_range * gap_frac * (level + 1.5))

# ── Fetch runs from W&B ────────────────────────────────────────────────────────

print(f"Fetching runs from: {SWEEP}")
api  = wandb.Api()
runs = api.runs(SWEEP)

records = []
for run in runs:
    summary = run.summary._json_dict
    config  = {k: v for k, v in run.config.items() if not k.startswith("_")}

    n_special   = summary.get("n_special_features")
    ce_increase = summary.get("ce_increase")
    exp_var     = summary.get("explained_variance")

    # Skip runs missing any of the three key metrics
    if any(v is None for v in [n_special, ce_increase, exp_var]):
        continue

    records.append({
        "run_name":      run.name,
        "state":         run.state,
        "n_special":     int(n_special),
        "ce_increase":   float(ce_increase),
        "explained_var": float(exp_var),
        # config columns kept for reference
        "d_sae":         config.get("d_sae"),
        "top_k":         config.get("top_k"),
        "lr":            config.get("lr"),
        "seed":          config.get("seed"),
        "sae_type":      config.get("sae_type"),
    })

df = pd.DataFrame(records)
print(f"Runs with complete data: {len(df)}")

if df.empty:
    print("No runs with complete data found. Exiting.")
    sys.exit(1)

# ── Bin n_special ──────────────────────────────────────────────────────────────

CATEGORY_ORDER = ["0", "1", "2", ">2"]

def bin_special(n):
    if n == 0: return "0"
    if n == 1: return "1"
    if n == 2: return "2"
    return ">2"

df["n_special_bin"] = pd.Categorical(
    df["n_special"].apply(bin_special),
    categories=CATEGORY_ORDER,
    ordered=True,
)

print("\nDistribution of n_special_features:")
print(df["n_special_bin"].value_counts().sort_index().to_string())

# ── Correlations ───────────────────────────────────────────────────────────────

corr_ce, pval_ce = spearmanr(df["n_special"], df["ce_increase"])
corr_ev, pval_ev = spearmanr(df["n_special"], df["explained_var"])

# Compute CIs for Spearman correlations
ci_ce = spearman_ci(corr_ce, len(df))
ci_ev = spearman_ci(corr_ev, len(df))

print("\nSpearman correlations (n_special_features vs metric):")
print(f"  CE increase:    r = {corr_ce:+.3f}  [{ci_ce[0]:+.3f}, {ci_ce[1]:+.3f}]  (p = {pval_ce:.3g})")
print(f"  Explained var:  r = {corr_ev:+.3f}  [{ci_ev[0]:+.3f}, {ci_ev[1]:+.3f}]  (p = {pval_ev:.3g})")

# ── Prepare CI data for error bars ────────────────────────────────────────────

ci_data_ce = {}
ci_data_ev = {}

for cat in CATEGORY_ORDER:
    df_cat = df[df["n_special_bin"] == cat]
    if not df_cat.empty:
        ci_data_ce[cat] = compute_cis_for_category(df_cat, "ce_increase", args.ci_method)
        ci_data_ev[cat] = compute_cis_for_category(df_cat, "explained_var", args.ci_method)

# ── Prepare p-value data for annotations ──────────────────────────────────────

pval_data_ce = pairwise_comparisons(df, "n_special_bin", "ce_increase")
pval_data_ev = pairwise_comparisons(df, "n_special_bin", "explained_var")

# ── Plotting Helper Function ───────────────────────────────────────────────────

def create_plot(include_ci=False, include_pvalues=False):
    """
    Create and save a plot with specified statistical annotations.
    
    Args:
        include_ci: whether to add confidence interval error bars
        include_pvalues: whether to add p-value significance brackets
    
    Returns:
        (fig, filename_base) where filename_base is used for saving
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    PALETTE = sns.color_palette("muted", n_colors=len(CATEGORY_ORDER))

    # hue= mirrors x= to satisfy the new seaborn API; legend=False suppresses the redundant legend
    STRIP_KW = dict(
        hue="n_special_bin", hue_order=CATEGORY_ORDER,
        palette=PALETTE, legend=False,
        size=3.5, alpha=0.5, jitter=True,
    )
    BOX_KW = dict(
        hue="n_special_bin", hue_order=CATEGORY_ORDER,
        palette=PALETTE, legend=False,
        width=0.45,
        boxprops=dict(alpha=0.35),
        whiskerprops=dict(alpha=0.45),
        capprops=dict(alpha=0.45),
        medianprops=dict(color="black", linewidth=1.8),
        fliersize=0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Special latent count vs SAE performance\n"
        + f"CE increase: r = {corr_ce:+.3f} [{ci_ce[0]:+.3f}, {ci_ce[1]:+.3f}], p = {pval_ce:.3g}  |  "
        + f"Explained var: r = {corr_ev:+.3f} [{ci_ev[0]:+.3f}, {ci_ev[1]:+.3f}], p = {pval_ev:.3g}",
        fontsize=12,
        y=1.02,
    )

    # Panel 1: CE increase (lower = better)
    ax = axes[0]
    sns.boxplot(data=df, x="n_special_bin", y="ce_increase",
                order=CATEGORY_ORDER, ax=ax, **BOX_KW)
    sns.stripplot(data=df, x="n_special_bin", y="ce_increase",
                  order=CATEGORY_ORDER, ax=ax, **STRIP_KW)
    ax.set_xlabel("Number of special latents", labelpad=8)
    ax.set_ylabel("CE increase (patched \u2212 baseline)")
    ax.set_title("CE increase  \u2193 better")
    ymin = ax.get_ylim()[0]
    for i, cat in enumerate(CATEGORY_ORDER):
        n = (df["n_special_bin"] == cat).sum()
        ax.text(i, ymin, f"n={n}", ha="center", va="bottom", fontsize=8.5, color="grey")

    # Add error bars if requested
    if include_ci:
        for i, cat in enumerate(CATEGORY_ORDER):
            if cat in ci_data_ce:
                ci_info = ci_data_ce[cat]
                if args.ci_method == "bootstrap" or args.ci_method == "both":
                    lower = ci_info.get("bootstrap_lower", np.mean(df[df["n_special_bin"] == cat]["ce_increase"]))
                    upper = ci_info.get("bootstrap_upper", np.mean(df[df["n_special_bin"] == cat]["ce_increase"]))
                else:  # sem
                    lower = ci_info.get("sem_lower", np.mean(df[df["n_special_bin"] == cat]["ce_increase"]))
                    upper = ci_info.get("sem_upper", np.mean(df[df["n_special_bin"] == cat]["ce_increase"]))
                
                ax.plot([i, i], [lower, upper], color="black", linewidth=2, alpha=0.7)
                ax.plot([i-0.1, i+0.1], [lower, lower], color="black", linewidth=2, alpha=0.7)
                ax.plot([i-0.1, i+0.1], [upper, upper], color="black", linewidth=2, alpha=0.7)

    # Add p-value annotations if requested
    if include_pvalues:
        COMPARISONS = [(0, 1, "0_vs_1"), (0, 2, "0_vs_2"), (0, 3, "0_vs_>2")]
        add_significance_brackets(axes[0], COMPARISONS, pval_data_ce)

    sns.despine(ax=ax)

    # Panel 2: Explained variance (higher = better)
    ax = axes[1]
    sns.boxplot(data=df, x="n_special_bin", y="explained_var",
                order=CATEGORY_ORDER, ax=ax, **BOX_KW)
    sns.stripplot(data=df, x="n_special_bin", y="explained_var",
                  order=CATEGORY_ORDER, ax=ax, **STRIP_KW)
    ax.set_xlabel("Number of special latents", labelpad=8)
    ax.set_ylabel("Explained variance")
    ax.set_title("Explained variance  \u2191 better")
    ymin = ax.get_ylim()[0]
    for i, cat in enumerate(CATEGORY_ORDER):
        n = (df["n_special_bin"] == cat).sum()
        ax.text(i, ymin, f"n={n}", ha="center", va="bottom", fontsize=8.5, color="grey")

    # Add error bars if requested
    if include_ci:
        for i, cat in enumerate(CATEGORY_ORDER):
            if cat in ci_data_ev:
                ci_info = ci_data_ev[cat]
                if args.ci_method == "bootstrap" or args.ci_method == "both":
                    lower = ci_info.get("bootstrap_lower", np.mean(df[df["n_special_bin"] == cat]["explained_var"]))
                    upper = ci_info.get("bootstrap_upper", np.mean(df[df["n_special_bin"] == cat]["explained_var"]))
                else:  # sem
                    lower = ci_info.get("sem_lower", np.mean(df[df["n_special_bin"] == cat]["explained_var"]))
                    upper = ci_info.get("sem_upper", np.mean(df[df["n_special_bin"] == cat]["explained_var"]))
                
                ax.plot([i, i], [lower, upper], color="black", linewidth=2, alpha=0.7)
                ax.plot([i-0.1, i+0.1], [lower, lower], color="black", linewidth=2, alpha=0.7)
                ax.plot([i-0.1, i+0.1], [upper, upper], color="black", linewidth=2, alpha=0.7)

    # Add p-value annotations if requested
    if include_pvalues:
        COMPARISONS = [(0, 1, "0_vs_1"), (0, 2, "0_vs_2"), (0, 3, "0_vs_>2")]
        add_significance_brackets(axes[1], COMPARISONS, pval_data_ev)

    sns.despine(ax=ax)

    plt.tight_layout()

    # Build filename suffix based on flags
    filename_parts = ["special_latents_vs_performance"]
    if include_ci:
        filename_parts.append("with_ci")
    if include_pvalues:
        filename_parts.append("with_pvalues")
    filename_base = "_".join(filename_parts)

    return fig, filename_base


# ── Generate all four visualization variants ────────────────────────────────────

variants = [
    (False, False, "baseline"),
    (True, False, "with CI error bars"),
    (False, True, "with p-value brackets"),
    (True, True, "with CI error bars and p-value brackets"),
]

for include_ci, include_pvalues, description in variants:
    fig, filename_base = create_plot(include_ci=include_ci, include_pvalues=include_pvalues)
    
    out_pdf = SAVE_DIR / f"{filename_base}.pdf"
    out_png = SAVE_DIR / f"{filename_base}.png"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {description:40s} → {filename_base}")
    
    plt.close(fig)

# ── Pairwise Comparisons ───────────────────────────────────────────────────────

print("\n" + "="*70)
print("Pairwise Mann-Whitney U tests (Category 0 vs all others):")
print("="*70)

for metric in ["ce_increase", "explained_var"]:
    print(f"\n{metric}:")
    comparisons = pairwise_comparisons(df, "n_special_bin", metric)
    for comp_name, (stat, pval) in comparisons.items():
        sig_marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  {comp_name:12s}  p = {pval:.4g}  {sig_marker}")

# ── Summary table ──────────────────────────────────────────────────────────────

print("\nPer-category summary (with confidence intervals):")

summary_rows = []

for cat in CATEGORY_ORDER:
    df_cat = df[df["n_special_bin"] == cat]
    if df_cat.empty:
        continue
    
    for metric in ["ce_increase", "explained_var"]:
        data = df_cat[metric].values
        
        row = {
            "Category": cat,
            "Metric": metric,
            "n": len(data),
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data, ddof=1),
        }
        
        # Add CIs based on method
        if args.ci_method in ["bootstrap", "both"]:
            bootstrap_lower, bootstrap_upper = bootstrap_ci(data)
            row["bootstrap_ci_lower"] = bootstrap_lower
            row["bootstrap_ci_upper"] = bootstrap_upper
        
        if args.ci_method in ["sem", "both"]:
            sem_lower, sem_upper = sem_ci(data)
            row["sem_ci_lower"] = sem_lower
            row["sem_ci_upper"] = sem_upper
        
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.round(4).to_string(index=False))

# Save summary table to markdown
out_md = SAVE_DIR / "special_latent_performance_summary.md"
with open(out_md, "w") as f:
    f.write("# Special Latent Performance Summary\n\n")
    f.write(summary_df.round(4).to_markdown(index=False))
    f.write("\n")
print(f"Summary table saved to {out_md}")

out_csv = SAVE_DIR / "special_latent_performance_data.csv"
df.to_csv(out_csv, index=False)
print(f"Raw data saved to {out_csv}")