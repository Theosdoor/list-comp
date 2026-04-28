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
from scipy.stats import spearmanr
import wandb

# ── Config ─────────────────────────────────────────────────────────────────────

SWEEP    = "theo-farrell99-durham-university/orderbyscale_sae_sweep"
SAVE_DIR = Path(__file__).parent.parent.parent / "results" / "special_latent_analysis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
sns.despine(ax=ax)

plt.tight_layout()

out_pdf = SAVE_DIR / "special_latents_vs_performance.pdf"
out_png = SAVE_DIR / "special_latents_vs_performance.png"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved figures to:\n  {out_pdf}\n  {out_png}")

plt.show()

# ── Summary table ──────────────────────────────────────────────────────────────

print("\nPer-category summary:")
summary = (
    df.groupby("n_special_bin", observed=True)[["ce_increase", "explained_var"]]
    .agg(["median", "mean", "std", "count"])
    .round(4)
)
print(summary.to_string())

out_csv = SAVE_DIR / "special_latent_performance_data.csv"
df.to_csv(out_csv, index=False)
print(f"\nRaw data saved to {out_csv}")