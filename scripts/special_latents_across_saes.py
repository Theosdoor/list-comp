#!/usr/bin/env python3
"""
SAE analysis plots: loss recovered and explained variance vs L0.

Produces plots and markdown reports per run (suffix includes r threshold):

  Per-type scatters (4):
    {type}_{metric}_r{thresh}.pdf
      x = actual L0, y = metric, hue = n_special_bin, fitted line per bin

  Per-type box+strip per metric (4):
    {type}_{metric}_boxplot_r{thresh}.pdf
      2x2 grid: col0=x:L0(integer-binned) hue:n_special,
                col1=x:n_special_bin hue:n_special

  All-types scatters (2):
    all_{metric}_r{thresh}.pdf
      2 panels: left hue=sae_type, right hue=n_special_bin, fitted line per hue

  All-types box+strip per metric (2):
    all_{metric}_boxplot_r{thresh}.pdf
      2x2 grid: col0=x:L0(integer-binned) hue:n_special,
                col1=x:n_special_bin hue:n_special

  Markdown reports:
    analysis_report_all_r{thresh}.md
    analysis_report_{type}_r{thresh}.md (one per SAE type present)

Metrics:
  loss_recovered  = (H* - H0) / (H_orig - H0)   [higher = better]
  explained_var   = 1 - MSE(recon) / Var(orig)   [higher = better]

Examples:
    python scripts/special_latents_across_saes.py \\
        --sae_dirs sae_checkpoints/ \\
        --model_path models/2layer_100dig_64d.pt

    python scripts/special_latents_across_saes.py \\
        --sae_dirs sae_checkpoints/btk sae_checkpoints/jumprelu \\
        --model_path models/2layer_100dig_64d.pt \\
        --alpha_diff_thresh 0.3 \\
        --output_dir results/plots \\
        --batch_size 1024
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.models.transformer import make_model
from src.models.utils import infer_model_config
from src.utils.runtime import configure_runtime
from src.data.datasets import get_dataset
from src.sae.loading import instantiate_sae_from_cfg, select_checkpoints
from src.sae.activation_collection import (
    collect_sae_activations,
    collect_attention_patterns,
    identify_special_features,
)
from src.sae.hooks import _encode_through_sae
from src.sae.metrics import compute_reconstruction_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CATEGORY_ORDER = ["0", "1", "2", ">2"]

METRICS = {
    "loss_recovered": ("Loss recovered",     "higher = better"),
    "explained_var":  ("Explained variance", "higher = better"),
}

TYPE_PALETTE  = {"btk": "#4c72b0", "jumprelu": "#dd8452", "matryoshka": "#55a868"}
TYPE_MARKERS  = {"btk": "o",       "jumprelu": "s",       "matryoshka": "^"}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(dirs: list[str]) -> list:
    checkpoints = []
    for d in dirs:
        root = Path(d) if not isinstance(d, Path) else d
        if root.is_file() and root.suffix == ".pt":
            checkpoints.append(root)
        elif root.is_dir():
            checkpoints.extend(sorted(root.rglob("*.pt")))
        else:
            print(f"  Warning: {d} is not a .pt file or directory — skipping.")
    return checkpoints


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_ce_passes(model, sae, act_mean, val_dl, sep_idx, list_len, device):
    """
    Three forward passes: baseline, SAE-patched, zero-ablated.
    Returns (h_orig, h_star, h0) as per-token mean CE values.
    """
    hook_name = "blocks.0.hook_resid_post"

    def sae_patch_hook(activations, hook):
        activations = activations.clone()
        sep_acts = activations[:, sep_idx, :]
        reconstructed = _encode_through_sae(sep_acts, sae, act_mean, decode=True)
        activations[:, sep_idx, :] = reconstructed + act_mean.to(reconstructed.device)
        return activations

    def zero_ablate_hook(activations, hook):
        activations = activations.clone()
        activations[:, sep_idx, :] = 0.0
        return activations

    h_orig_sum = h_star_sum = h0_sum = 0.0
    n_tokens = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="    CE passes", leave=False):
            inputs   = inputs.to(device)
            targets  = targets.to(device)
            out_tgt  = targets[:, list_len + 1:]
            b, t     = out_tgt.shape

            orig_logits = model(inputs)[:, list_len + 1:]
            v = orig_logits.shape[-1]

            h_orig_sum += F.cross_entropy(
                orig_logits.reshape(b * t, v), out_tgt.reshape(b * t), reduction="sum"
            ).item()

            star_logits = model.run_with_hooks(
                inputs, fwd_hooks=[(hook_name, sae_patch_hook)]
            )[:, list_len + 1:]
            h_star_sum += F.cross_entropy(
                star_logits.reshape(b * t, v), out_tgt.reshape(b * t), reduction="sum"
            ).item()

            zero_logits = model.run_with_hooks(
                inputs, fwd_hooks=[(hook_name, zero_ablate_hook)]
            )[:, list_len + 1:]
            h0_sum += F.cross_entropy(
                zero_logits.reshape(b * t, v), out_tgt.reshape(b * t), reduction="sum"
            ).item()

            n_tokens += b * t

    h_orig = h_orig_sum / n_tokens
    h_star = h_star_sum / n_tokens
    h0     = h0_sum     / n_tokens
    return h_orig, h_star, h0


def evaluate_sae(model, sae, act_mean, val_dl, sep_idx, list_len,
                 alpha_diff_thresh, device):
    """Full evaluation for one SAE checkpoint."""
    _, _, sae_acts_all = collect_sae_activations(
        model, sae, val_dl, act_mean,
        layer_idx=0, sep_idx=sep_idx, device=device,
    )
    actual_l0 = (sae_acts_all > 0).float().sum(dim=1).mean().item()

    h_orig, h_star, h0 = compute_ce_passes(
        model, sae, act_mean, val_dl, sep_idx, list_len, device
    )
    denom = h_orig - h0
    loss_recovered = (h_star - h0) / denom if abs(denom) > 1e-8 else float("nan")

    recon = compute_reconstruction_metrics(
        model, sae, val_dl, act_mean, layer_idx=0, sep_idx=sep_idx, device=device,
    )
    explained_var = recon["explained_variance"]

    alpha_d1_all, alpha_d2_all = collect_attention_patterns(
        model, val_dl, layer_idx=0, sep_idx=sep_idx, device=device,
    )
    special_info = identify_special_features(
        sae_acts_all, alpha_d1_all, alpha_d2_all, threshold=alpha_diff_thresh,
    )

    return {
        "actual_l0":      actual_l0,
        "loss_recovered": loss_recovered,
        "explained_var":  explained_var,
        "h_orig":         h_orig,
        "h_star":         h_star,
        "h0":             h0,
        "n_special":      special_info["n_special_features"],
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def bin_special(n: int) -> str:
    if n < 0:
        raise ValueError(f"n_special must be >= 0, got {n}")
    return {0: "0", 1: "1", 2: "2"}.get(n, ">2")


def _fit_line(x, y):
    """Linear fit; returns (x_grid, y_hat) or None if fewer than 2 points."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    coeffs = np.polyfit(x[mask], y[mask], 1)
    xg = np.linspace(x[mask].min(), x[mask].max(), 100)
    return xg, np.polyval(coeffs, xg)


def _add_fitted_lines(ax, df_sub, x_col, y_col, hue_col, palette, order):
    """Overlay one fitted line per hue category."""
    for cat in order:
        sub = df_sub[df_sub[hue_col] == cat]
        if len(sub) < 2:
            continue
        result = _fit_line(sub[x_col].values.astype(float),
                           sub[y_col].values.astype(float))
        if result is None:
            continue
        xg, yh = result
        ax.plot(xg, yh, color=palette[cat], linewidth=1.4, linestyle="--", alpha=0.8)


# ── Per-type scatter ──────────────────────────────────────────────────────────

def plot_per_type_scatter(df_type, sae_type, metric_key, thresh, output_dir):
    ylabel, direction = METRICS[metric_key]
    n_cats   = len(CATEGORY_ORDER)
    palette  = dict(zip(CATEGORY_ORDER, sns.color_palette("muted", n_colors=n_cats)))
    present  = [c for c in CATEGORY_ORDER if c in df_type["n_special_bin"].values]

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(
        data=df_type, x="actual_l0", y=metric_key,
        hue="n_special_bin", hue_order=present,
        palette={c: palette[c] for c in present},
        s=55, alpha=0.75, ax=ax,
    )
    _add_fitted_lines(ax, df_type, "actual_l0", metric_key,
                      "n_special_bin", palette, present)

    ax.set_xlabel("Actual L0 (mean active features per token)")
    ax.set_ylabel(ylabel)
    sae_type_label = sae_type if sae_type in TYPE_PALETTE else f"{sae_type} (unknown)"
    ax.set_title(f"{sae_type_label} — {ylabel}\n(r threshold = {thresh})")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=f"Special latents\n(|r| > {thresh})",
              loc="best", framealpha=0.9)

    sns.despine(ax=ax)
    plt.tight_layout()
    out = output_dir / f"{sae_type}_{metric_key}_r{thresh}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── Per-type box + strip ──────────────────────────────────────────────────────

def plot_per_type_boxplot(df_type, sae_type, metric_key, thresh, output_dir):
    """Create a single 1x2 box+strip plot for one metric."""
    n_cats  = len(CATEGORY_ORDER)
    palette = dict(zip(CATEGORY_ORDER, sns.color_palette("muted", n_colors=n_cats)))
    present = [c for c in CATEGORY_ORDER if c in df_type["n_special_bin"].values]

    # Integer-bin L0 for the left column
    df_type = df_type.copy()
    df_type["l0_bin"] = df_type["actual_l0"].round().astype(int).astype(str)
    l0_order = [str(v) for v in sorted(df_type["l0_bin"].astype(int).unique())]

    ylabel, direction = METRICS[metric_key]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{sae_type} SAE — {ylabel} by L0 and special latent count  (|r| > {thresh})",
        fontsize=12, y=1.02,
    )

    BOX_KW = dict(
        hue="n_special_bin", hue_order=present,
        palette={c: palette[c] for c in present},
        legend=False, width=0.55,
        boxprops=dict(alpha=0.35),
        whiskerprops=dict(alpha=0.45),
        capprops=dict(alpha=0.45),
        medianprops=dict(color="black", linewidth=1.8),
        fliersize=0,
    )
    STRIP_KW = dict(
        hue="n_special_bin", hue_order=present,
        palette={c: palette[c] for c in present},
        legend=False, size=3.0, alpha=0.5, jitter=True,
    )

    # --- Col 0: x = L0 bin, hue = n_special_bin ---
    ax = axes[0]
    sns.boxplot(data=df_type, x="l0_bin", y=metric_key,
                order=l0_order, ax=ax, **BOX_KW)
    sns.stripplot(data=df_type, x="l0_bin", y=metric_key,
                  order=l0_order, ax=ax, **STRIP_KW)
    ax.set_xlabel("L0 (integer-binned)", labelpad=6)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  ({direction})")
    ymin = ax.get_ylim()[0]
    for i, lb in enumerate(l0_order):
        n = (df_type["l0_bin"] == lb).sum()
        ax.text(i, ymin, f"n={n}", ha="center", va="bottom",
                fontsize=7.5, color="grey")

    # legend on left plot
    handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=palette[c], markersize=8, label=c)
        for c in present
    ]
    ax.legend(handles=handles, title=f"Special latents\n(|r| > {thresh})",
              framealpha=0.9, fontsize=8)
    sns.despine(ax=ax)

    # --- Col 1: x = n_special_bin, hue = n_special_bin ---
    ax = axes[1]
    sns.boxplot(data=df_type, x="n_special_bin", y=metric_key,
                order=present, ax=ax, **BOX_KW)
    sns.stripplot(data=df_type, x="n_special_bin", y=metric_key,
                  order=present, ax=ax, **STRIP_KW)
    ax.set_xlabel(f"Special latents (|r| > {thresh})", labelpad=6)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  ({direction})")
    ymin = ax.get_ylim()[0]
    for i, cat in enumerate(present):
        n = (df_type["n_special_bin"] == cat).sum()
        ax.text(i, ymin, f"n={n}", ha="center", va="bottom",
                fontsize=7.5, color="grey")
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / f"{sae_type}_{metric_key}_boxplot_r{thresh}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── All-types scatter ─────────────────────────────────────────────────────────

def plot_all_types_scatter(df, metric_key, thresh, output_dir):
    ylabel, direction = METRICS[metric_key]
    n_cats   = len(CATEGORY_ORDER)
    cat_palette = dict(zip(CATEGORY_ORDER, sns.color_palette("muted", n_colors=n_cats)))
    present_cats = [c for c in CATEGORY_ORDER if c in df["n_special_bin"].values]
    present_types = [t for t in df["sae_type"].unique() if t in TYPE_PALETTE]
    unknown_types = [t for t in df["sae_type"].unique() if t not in TYPE_PALETTE]
    if unknown_types:
        print(f"    ⚠ Skipping plotting for unknown types: {unknown_types}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"All SAE types — {ylabel}  (|r| > {thresh})",
        fontsize=12,
    )

    # Panel 1: hue = sae_type
    ax = axes[0]
    for sae_type in present_types:
        sub = df[df["sae_type"] == sae_type]
        ax.scatter(
            sub["actual_l0"], sub[metric_key],
            color=TYPE_PALETTE[sae_type],
            marker=TYPE_MARKERS[sae_type],
            s=45, alpha=0.65, label=sae_type,
        )
        result = _fit_line(sub["actual_l0"].values.astype(float),
                           sub[metric_key].values.astype(float))
        if result is not None:
            xg, yh = result
            ax.plot(xg, yh, color=TYPE_PALETTE[sae_type],
                    linewidth=1.4, linestyle="--", alpha=0.85)

    ax.set_xlabel("Actual L0 (mean active features per token)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Hue = SAE type  ({direction})")
    ax.legend(title="SAE type", framealpha=0.9)
    sns.despine(ax=ax)

    # Panel 2: hue = n_special_bin
    ax = axes[1]
    for cat in present_cats:
        sub = df[df["n_special_bin"] == cat]
        ax.scatter(
            sub["actual_l0"], sub[metric_key],
            color=cat_palette[cat],
            s=45, alpha=0.65, label=cat,
        )
        result = _fit_line(sub["actual_l0"].values.astype(float),
                           sub[metric_key].values.astype(float))
        if result is not None:
            xg, yh = result
            ax.plot(xg, yh, color=cat_palette[cat],
                    linewidth=1.4, linestyle="--", alpha=0.85)

    ax.set_xlabel("Actual L0 (mean active features per token)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Hue = special latents  ({direction})")
    ax.legend(title=f"Special latents\n(|r| > {thresh})", framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / f"all_{metric_key}_r{thresh}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── All-types box + strip ─────────────────────────────────────────────────────

def plot_all_types_boxplot(df, metric_key, thresh, output_dir):
    """Create a single 1x2 box+strip plot for one metric across all SAEs."""
    n_cats  = len(CATEGORY_ORDER)
    palette = dict(zip(CATEGORY_ORDER, sns.color_palette("muted", n_colors=n_cats)))
    present = [c for c in CATEGORY_ORDER if c in df["n_special_bin"].values]

    # Integer-bin L0
    df = df.copy()
    df["l0_bin"] = df["actual_l0"].round().astype(int).astype(str)
    l0_order = [str(v) for v in sorted(df["l0_bin"].astype(int).unique())]

    ylabel, direction = METRICS[metric_key]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"All SAE types — {ylabel} by L0 and special latent count  (|r| > {thresh})",
        fontsize=12, y=1.02,
    )

    BOX_KW = dict(
        hue="n_special_bin", hue_order=present,
        palette={c: palette[c] for c in present},
        legend=False, width=0.55,
        boxprops=dict(alpha=0.35),
        whiskerprops=dict(alpha=0.45),
        capprops=dict(alpha=0.45),
        medianprops=dict(color="black", linewidth=1.8),
        fliersize=0,
    )
    STRIP_KW = dict(
        hue="n_special_bin", hue_order=present,
        palette={c: palette[c] for c in present},
        legend=False, size=3.0, alpha=0.5, jitter=True,
    )

    # --- Col 0: x = L0 bin, hue = n_special_bin ---
    ax = axes[0]
    sns.boxplot(data=df, x="l0_bin", y=metric_key,
                order=l0_order, ax=ax, **BOX_KW)
    sns.stripplot(data=df, x="l0_bin", y=metric_key,
                  order=l0_order, ax=ax, **STRIP_KW)
    ax.set_xlabel("L0 (integer-binned)", labelpad=6)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  ({direction})")
    ymin = ax.get_ylim()[0]
    for i, lb in enumerate(l0_order):
        n = (df["l0_bin"] == lb).sum()
        ax.text(i, ymin, f"n={n}", ha="center", va="bottom",
                fontsize=7.5, color="grey")

    # legend on left plot
    handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=palette[c], markersize=8, label=c)
        for c in present
    ]
    ax.legend(handles=handles, title=f"Special latents\n(|r| > {thresh})",
              framealpha=0.9, fontsize=8)
    sns.despine(ax=ax)

    # --- Col 1: x = n_special_bin, hue = n_special_bin ---
    ax = axes[1]
    sns.boxplot(data=df, x="n_special_bin", y=metric_key,
                order=present, ax=ax, **BOX_KW)
    sns.stripplot(data=df, x="n_special_bin", y=metric_key,
                  order=present, ax=ax, **STRIP_KW)
    ax.set_xlabel(f"Special latents (|r| > {thresh})", labelpad=6)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  ({direction})")
    ymin = ax.get_ylim()[0]
    for i, cat in enumerate(present):
        n = (df["n_special_bin"] == cat).sum()
        ax.text(i, ymin, f"n={n}", ha="center", va="bottom",
                fontsize=7.5, color="grey")
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / f"all_{metric_key}_boxplot_r{thresh}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Markdown Report Generation
# ---------------------------------------------------------------------------

def _define_l0_ranges(df: pd.DataFrame):
    """
    Define L0 ranges corresponding to integer bins.
    E.g., [0, 1.5), [1.5, 2.5), [2.5, 3.5), etc.
    First bin is clamped to 0 (not -0.5); rest are integer bins.
    Returns list of (min_val, max_val, label) tuples.
    """
    l0_max = df["actual_l0"].max()
    max_bin = int(np.ceil(l0_max))
    
    ranges = []
    # First bin: 0 to 1.5 (clamped to 0)
    ranges.append((0.0, 1.5, "0<L0<1.5"))
    
    # Subsequent bins: 1.5 to 2.5, 2.5 to 3.5, etc.
    for bin_idx in range(2, max_bin + 2):
        min_val = bin_idx - 0.5
        max_val = bin_idx + 0.5
        label = f"{min_val}<L0<{max_val}"
        ranges.append((min_val, max_val, label))
    
    return ranges


def _assign_l0_range(l0_val: float, ranges: list) -> str:
    """Assign a single L0 value to its range label."""
    for min_val, max_val, label in ranges:
        if min_val <= l0_val < max_val:
            return label
    # If value exceeds all ranges, check if it's beyond the last range's upper bound
    if ranges and l0_val >= ranges[-1][1]:
        return ranges[-1][2]
    raise ValueError(f"L0 value {l0_val} does not fall within any defined range: {ranges}")


def _format_cell(count: int, mean_val: float, std_err: float) -> str:
    """Format a cell with count | mean ± error."""
    if count == 0:
        return "—"
    if pd.isna(mean_val) or pd.isna(std_err):
        return f"{count} | {mean_val:.3f}±N/A"
    return f"{count} | {mean_val:.3f}±{std_err:.3f}"


def generate_markdown_report(df: pd.DataFrame, thresh: float, metric_key: str = "loss_recovered"):
    """
    Generate a markdown report with L0 ranges vs n_special_bin.
    Each cell shows: count / mean_metric ± std_err
    """
    metric_name = "Loss Recovered" if metric_key == "loss_recovered" else "Explained Variance"
    
    # Define L0 ranges
    ranges = _define_l0_ranges(df)
    
    # Add L0 range label to dataframe
    df = df.copy()
    df["l0_range"] = df["actual_l0"].apply(lambda x: _assign_l0_range(x, ranges))
    
    # Create markdown
    md_lines = [
        f"## {metric_name} vs L0 and Special Latent Count",
        "",
    ]
    
    # Table header
    md_lines.append("| L0 Range | All SAEs | 0 Special | 1 Special | 2 Special | >2 Special |")
    md_lines.append("|----------|----------|-----------|-----------|-----------|------------|")
    
    # Determine format based on metric
    sig_figs = 4 if metric_key == "loss_recovered" else 3
    
    # Table rows
    for min_val, max_val, l0_label in ranges:
        df_range = df[df["l0_range"] == l0_label]
        if df_range.empty:
            continue
        
        all_count = len(df_range)
        row_cells = [l0_label, str(all_count)]
        
        for n_special_cat in CATEGORY_ORDER:
            df_cell = df_range[df_range["n_special_bin"] == n_special_cat]
            count = len(df_cell)
            
            if count == 0:
                row_cells.append("—")
            else:
                mean_val = df_cell[metric_key].mean()
                std_err = df_cell[metric_key].sem()  # Standard error of the mean
                if pd.isna(mean_val) or pd.isna(std_err) or std_err == 0.0:
                    # Format with sig figs: use g format
                    formatted_mean = f"{mean_val:.{sig_figs-1}g}"
                    row_cells.append(f"{count} / {formatted_mean}")
                else:
                    # Format both mean and std_err with sig figs
                    formatted_mean = f"{mean_val:.{sig_figs-1}g}"
                    formatted_err = f"{std_err:.{sig_figs-1}g}"
                    row_cells.append(f"{count} / {formatted_mean}±{formatted_err}")
        
        md_lines.append("| " + " | ".join(row_cells) + " |")
    
    md_lines.append("")
    
    return "\n".join(md_lines)


def _slugify_sae_type(sae_type: str) -> str:
    """Convert SAE type to a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", sae_type.strip())
    return slug.strip("_") or "unknown"


def _build_markdown_report(df: pd.DataFrame, thresh: float, scope_label: str) -> str:
    """Build the full markdown report content for a dataframe subset."""
    md_content = f"# SAE Analysis Report ({scope_label}, |r| > {thresh})\n\n"
    md_content += generate_markdown_report(df, thresh, "loss_recovered")
    md_content += "\n---\n\n"
    md_content += generate_markdown_report(df, thresh, "explained_var")
    return md_content


def save_markdown_report(df: pd.DataFrame, thresh: float, output_dir: Path):
    """Save one all-types markdown report and one report per SAE type."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_report_path = output_dir / f"analysis_report_all_r{thresh}.md"
    all_report_path.write_text(_build_markdown_report(df, thresh, "all SAE types"))
    print(f"✓ Markdown report saved to {all_report_path}")

    for sae_type, df_type in df.groupby("sae_type", sort=True):
        type_slug = _slugify_sae_type(sae_type)
        type_report_path = output_dir / f"analysis_report_{type_slug}_r{thresh}.md"
        type_report_path.write_text(_build_markdown_report(df_type, thresh, f"sae_type={sae_type}"))
        print(f"✓ Markdown report saved to {type_report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAE analysis plots — loss recovered and explained variance vs L0.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sae_dirs", nargs="+", required=True,
                        help="Directories (or .pt files) containing SAE checkpoints.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the transformer checkpoint.")
    parser.add_argument("--alpha_diff_thresh", type=float, default=0.5,
                        help="Correlation threshold for special latent detection (default: 0.5).")
    parser.add_argument("--output_dir", type=str, default="results/sae_plots",
                        help="Directory for output PDFs and metrics CSV (default: results/sae_plots).")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for forward passes (default: 512).")
    parser.add_argument("--exclude_l0", type=int, nargs="+", default=None,
                        metavar="L0",
                        help="Integer L0 values to exclude from plots (e.g. --exclude_l0 1 2).")
    parser.add_argument("--exclude_d_sae", type=int, nargs="+", default=None,
                        metavar="D_SAE",
                        help="Dictionary sizes to exclude (e.g. --exclude_d_sae 64 128).")
    parser.add_argument("--exclude_type", type=str, nargs="+", default=None,
                        metavar="TYPE",
                        help="SAE types to exclude (e.g. --exclude_type jumprelu).")
    args = parser.parse_args()

    thresh = args.alpha_diff_thresh

    # --- Model ---
    model_cfg = infer_model_config(args.model_path)
    n_layers  = model_cfg["n_layers"]
    n_heads   = model_cfg["n_heads"]
    d_model   = model_cfg["d_model"]
    n_digits  = model_cfg["d_vocab"] - 2
    list_len  = model_cfg["list_len"]
    sep_idx   = list_len

    configure_runtime(
        list_len=list_len, seq_len=list_len * 2 + 1,
        vocab=n_digits + 2, device=DEVICE,
    )
    model = make_model(
        n_layers=n_layers, n_heads=n_heads, d_model=d_model,
        ln=model_cfg.get("use_ln", False),
        use_bias=model_cfg.get("use_bias", False),
        use_wv=model_cfg.get("use_wv", False),
        use_wo=model_cfg.get("use_wo", False),
    )
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f"✓ Loaded transformer: {args.model_path}  "
          f"(d_model={d_model}, n_digits={n_digits}, list_len={list_len})")

    # --- Data ---
    train_ds, val_ds = get_dataset(list_len=list_len, n_digits=n_digits, no_dupes=False)
    full_ds = ConcatDataset([train_ds, val_ds])
    val_dl = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)
    print(f"✓ Full dataset (train + val): {len(full_ds):,} samples\n")

    # --- Discover checkpoints ---
    all_checkpoints = find_checkpoints(args.sae_dirs)
    if not all_checkpoints:
        print("No .pt checkpoints found. Exiting.")
        sys.exit(1)
    
    # Filter to final checkpoints only (skip _best_ variants by default)
    checkpoints, using_best = select_checkpoints(all_checkpoints, use_best=False)
    if using_best:
        print(f"Found {len(all_checkpoints)} total checkpoints; using {len(checkpoints)} final "
              f"+ {len(using_best)} 'best'-only variants where no final available\n")
    else:
        print(f"Using {len(checkpoints)} final checkpoint(s)\n")

    # Pre-evaluation d_sae exclusion (mirrors --exclude-d-sae in compare_sae.py)
    if args.exclude_d_sae:
        def _d_sae_from_name(p):
            m = re.search(r'_d(\d+)_', Path(p).name)
            return int(m.group(1)) if m else None
        before = len(checkpoints)
        checkpoints = [p for p in checkpoints if _d_sae_from_name(p) not in args.exclude_d_sae]
        print(f"Excluded d_sae {args.exclude_d_sae}: {before} → {len(checkpoints)} checkpoints\n")

    # --- Evaluate ---
    records = []
    for i, ckpt_path in enumerate(checkpoints):
        print(f"[{i+1}/{len(checkpoints)}] {ckpt_path.name}")
        try:
            ckpt     = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            cfg      = ckpt.get("cfg", {})
            sae_type = cfg.get("sae_type", "btk")
            d_sae    = cfg.get("dict_size", cfg.get("d_sae"))
            
            # Validate d_sae
            if d_sae is None:
                print(f"  ✗ Failed: No dict_size or d_sae in checkpoint config")
                continue
            
            # Validate sae_type (warn if unknown, but don't fail)
            if sae_type not in TYPE_PALETTE:
                print(f"  ⚠ Warning: Unknown sae_type '{sae_type}' (expected: btk, jumprelu, matryoshka)")

            sae = instantiate_sae_from_cfg(cfg, d_model, DEVICE)
            sae.load_state_dict(ckpt["state_dict"])
            sae.eval()
            act_mean = ckpt.get("act_mean", torch.zeros(d_model)).to(DEVICE)

            metrics = evaluate_sae(
                model, sae, act_mean, val_dl,
                sep_idx, list_len, thresh, DEVICE,
            )
            records.append({
                "checkpoint": ckpt_path.name,
                "sae_type":   sae_type,
                "d_sae":      d_sae,
                **metrics,
            })
            print(
                f"  L0={metrics['actual_l0']:.2f}  "
                f"lr={metrics['loss_recovered']:.4f}  "
                f"ev={metrics['explained_var']:.4f}  "
                f"n_special={metrics['n_special']}"
            )
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            import traceback; traceback.print_exc()

    if not records:
        print("\nNo SAEs evaluated successfully. Exiting.")
        sys.exit(1)

    # --- DataFrame ---
    df = pd.DataFrame(records)
    df["n_special_bin"] = pd.Categorical(
        df["n_special"].apply(bin_special),
        categories=CATEGORY_ORDER, ordered=True,
    )

    # --- Apply exclusions to plot data (evaluation always runs on all checkpoints) ---
    df_plot = df.copy()
    n_before = len(df_plot)
    if args.exclude_l0:
        l0_int = df_plot["actual_l0"].round().astype(int)
        df_plot = df_plot[~l0_int.isin(args.exclude_l0)]
        print(f"  Excluded L0 {args.exclude_l0}: {n_before} → {len(df_plot)} SAEs")
        n_before = len(df_plot)
    if args.exclude_d_sae:
        df_plot = df_plot[~df_plot["d_sae"].isin(args.exclude_d_sae)]
        print(f"  Excluded d_sae {args.exclude_d_sae}: {n_before} → {len(df_plot)} SAEs")
        n_before = len(df_plot)
    if args.exclude_type:
        df_plot = df_plot[~df_plot["sae_type"].isin(args.exclude_type)]
        print(f"  Excluded types {args.exclude_type}: {n_before} → {len(df_plot)} SAEs")
    if df_plot.empty:
        print("\nAll SAEs filtered out by exclusion flags. Exiting.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    # Organize by alpha threshold
    plot_dir = output_dir / f"r{thresh}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save full (unfiltered) CSV so exclusions don't lose data
    csv_path = plot_dir / f"sae_metrics_r{thresh}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Full metrics saved to {csv_path}")

    # --- Generate plots (using filtered df_plot) ---
    print("\nGenerating plots...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    for sae_type, df_type in df_plot.groupby("sae_type"):
        df_type = df_type.copy()
        n = len(df_type)
        if n < 2:
            print(f"  ⚠ Skipping {sae_type}: only {n} checkpoint(s).")
            continue
        print(f"\n  {sae_type} ({n} SAEs):")

        for metric_key in METRICS:
            plot_per_type_scatter(df_type, sae_type, metric_key, thresh, plot_dir)
            plot_per_type_boxplot(df_type, sae_type, metric_key, thresh, plot_dir)

    print("\n  All-types:")
    for metric_key in METRICS:
        plot_all_types_scatter(df_plot, metric_key, thresh, plot_dir)
        plot_all_types_boxplot(df_plot, metric_key, thresh, plot_dir)

    # --- Generate markdown report ---
    print("\nGenerating markdown report...")
    save_markdown_report(df_plot, thresh, plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
