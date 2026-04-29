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
    description="Special Latent vs SAE Performance Analysis with optional statistical enhancements",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--ci",
    action="store_true",
    help="Add error bars showing 95%% confidence intervals to plots"
)
parser.add_argument(
    "--pvalues",
    action="store_true",
    help="Add p-value annotations for Mann-Whitney U tests (category 0 vs others)"
)
parser.add_argument(
    "--ci-method",
    choices=["bootstrap", "sem", "both"],
    default="bootstrap",
    help="Method for computing confidence intervals in summary table (default: bootstrap)"
)
parser.add_argument(
    "--reduced-table",
    action="store_true",
    help="Print only basic stats without confidence interval columns"
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

print("\nSpearman correlations (n_special_features vs metric):")
print(f"  CE increase:    r = {corr_ce:+.3f}  (p = {pval_ce:.3g})")
print(f"  Explained var:  r = {corr_ev:+.3f}  (p = {pval_ev:.3g})")

# ── Prepare CI data for optional error bars ────────────────────────────────────

ci_data_ce = {}
ci_data_ev = {}

if args.ci:
    for cat in CATEGORY_ORDER:
        df_cat = df[df["n_special_bin"] == cat]
        if not df_cat.empty:
            ci_data_ce[cat] = compute_cis_for_category(df_cat, "ce_increase", args.ci_method)
            ci_data_ev[cat] = compute_cis_for_category(df_cat, "explained_var", args.ci_method)

# ── Prepare p-value data for optional annotations ──────────────────────────────

pval_data_ce = {}
pval_data_ev = {}

if args.pvalues:
    pval_data_ce = pairwise_comparisons(df, "n_special_bin", "ce_increase")
    pval_data_ev = pairwise_comparisons(df, "n_special_bin", "explained_var")

# ── Plotting ───────────────────────────────────────────────────────────────────

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
    + f"(Spearman r:  CE increase = {corr_ce:+.3f},  Explained var = {corr_ev:+.3f})",
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

# Add error bars if --ci flag is set
if args.ci:
    for i, cat in enumerate(CATEGORY_ORDER):
        if cat in ci_data_ce:
            ci_info = ci_data_ce[cat]
            # Get the method-specific bounds
            if args.ci_method == "bootstrap" or args.ci_method == "both":
                lower = ci_info.get("bootstrap_lower", np.mean(df[df["n_special_bin"] == cat]["ce_increase"]))
                upper = ci_info.get("bootstrap_upper", np.mean(df[df["n_special_bin"] == cat]["ce_increase"]))
            else:  # sem
                lower = ci_info.get("sem_lower", np.mean(df[df["n_special_bin"] == cat]["ce_increase"]))
                upper = ci_info.get("sem_upper", np.mean(df[df["n_special_bin"] == cat]["ce_increase"]))
            
            # Plot error bar centered at x position i
            ax.plot([i, i], [lower, upper], color="black", linewidth=2, alpha=0.7)
            ax.plot([i-0.1, i+0.1], [lower, lower], color="black", linewidth=2, alpha=0.7)
            ax.plot([i-0.1, i+0.1], [upper, upper], color="black", linewidth=2, alpha=0.7)

# Add p-value annotations if --pvalues flag is set
if args.pvalues:
    y_max = ax.get_ylim()[1]
    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05  # 5% of range
    
    # Annotate comparisons: 0 vs 1, 0 vs 2, 0 vs >2
    comparisons = [
        (0, 1, "0_vs_1"),
        (0, 2, "0_vs_2"),
        (0, 3, "0_vs_>2"),
    ]
    
    for i, j, comp_key in comparisons:
        if comp_key in pval_data_ce:
            pval = pval_data_ce[comp_key][1]
            label = format_pvalue_label(pval)
            x_mid = (i + j) / 2.0
            ax.text(x_mid, y_max + y_offset, label, ha="center", fontsize=10, fontweight="bold")

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

# Add error bars if --ci flag is set
if args.ci:
    for i, cat in enumerate(CATEGORY_ORDER):
        if cat in ci_data_ev:
            ci_info = ci_data_ev[cat]
            # Get the method-specific bounds
            if args.ci_method == "bootstrap" or args.ci_method == "both":
                lower = ci_info.get("bootstrap_lower", np.mean(df[df["n_special_bin"] == cat]["explained_var"]))
                upper = ci_info.get("bootstrap_upper", np.mean(df[df["n_special_bin"] == cat]["explained_var"]))
            else:  # sem
                lower = ci_info.get("sem_lower", np.mean(df[df["n_special_bin"] == cat]["explained_var"]))
                upper = ci_info.get("sem_upper", np.mean(df[df["n_special_bin"] == cat]["explained_var"]))
            
            # Plot error bar centered at x position i
            ax.plot([i, i], [lower, upper], color="black", linewidth=2, alpha=0.7)
            ax.plot([i-0.1, i+0.1], [lower, lower], color="black", linewidth=2, alpha=0.7)
            ax.plot([i-0.1, i+0.1], [upper, upper], color="black", linewidth=2, alpha=0.7)

# Add p-value annotations if --pvalues flag is set
if args.pvalues:
    y_max = ax.get_ylim()[1]
    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05  # 5% of range
    
    # Annotate comparisons: 0 vs 1, 0 vs 2, 0 vs >2
    comparisons = [
        (0, 1, "0_vs_1"),
        (0, 2, "0_vs_2"),
        (0, 3, "0_vs_>2"),
    ]
    
    for i, j, comp_key in comparisons:
        if comp_key in pval_data_ev:
            pval = pval_data_ev[comp_key][1]
            label = format_pvalue_label(pval)
            x_mid = (i + j) / 2.0
            ax.text(x_mid, y_max + y_offset, label, ha="center", fontsize=10, fontweight="bold")

sns.despine(ax=ax)

plt.tight_layout()

# ── Generate Output Filenames ──────────────────────────────────────────────────

# Build filename suffix based on active flags
filename_parts = ["special_latents_vs_performance"]

if args.ci:
    filename_parts.append("with_ci")
if args.pvalues:
    filename_parts.append("with_pvalues")

filename_base = "_".join(filename_parts)

out_pdf = SAVE_DIR / f"{filename_base}.pdf"
out_png = SAVE_DIR / f"{filename_base}.png"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved figures to:\n  {out_pdf}\n  {out_png}")

plt.show()

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

print("\nPer-category summary:")

if args.reduced_table:
    # Original table format: basic stats only
    summary = (
        df.groupby("n_special_bin", observed=True)[["ce_increase", "explained_var"]]
        .agg(["median", "mean", "std", "count"])
        .round(4)
    )
    print(summary.to_string())
else:
    # Enhanced table: includes confidence intervals
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

out_csv = SAVE_DIR / "special_latent_performance_data.csv"
df.to_csv(out_csv, index=False)
print(f"\nRaw data saved to {out_csv}")