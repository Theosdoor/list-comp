"""
SAE Visualization Utilities

Functions for creating plots and visualizations of SAE features and activations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_feature_heatmaps(d1_all, d2_all, sae_acts_all, n_digits=100, figsize=(25, 25), skip_dead_latents=True, shared_scale=False):
    """
    Create interactive Plotly grid of heatmaps for all SAE features.

    Args:
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: Tensor of SAE activations [n_samples, d_sae]
        n_digits: Number of possible digit values
        figsize: Figure size (width, height) in inches
        skip_dead_latents: If True, omit features that never fire (default True)

    Returns:
        fig: Plotly Figure object (interactive)
    """
    d_sae = sae_acts_all.shape[1]
    feature_indices = [i for i in range(d_sae) if not (skip_dead_latents and sae_acts_all[:, i].sum() == 0)]
    n_active = len(feature_indices)
    all_act_matrices = _compute_act_matrices(d1_all, d2_all, sae_acts_all, feature_indices, n_digits)

    global_max = float(max(m.max() for m in all_act_matrices)) if shared_scale else None

    # Create subplot grid sized to active features only
    grid_size = int(np.ceil(np.sqrt(n_active)))

    subplot_titles = [f'Latent {i}' for i in feature_indices]
    total_subplots = grid_size * grid_size
    subplot_titles.extend([''] * (total_subplots - n_active))
    
    fig = make_subplots(
        rows=grid_size,
        cols=grid_size,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )
    
    # Add heatmaps
    for j, feat_idx in enumerate(feature_indices):
        row = j // grid_size + 1
        col = j % grid_size + 1

        fig.add_trace(
            go.Heatmap(
                z=all_act_matrices[j].numpy(),
                colorscale='Viridis',
                zmin=0 if shared_scale else None,
                zmax=global_max if shared_scale else None,
                showscale=(j == n_active - 1),
                hovertemplate='d1: %{x}<br>d2: %{y}<br>Activation: %{z:.4f}<extra></extra>',
                name=f'Latent {feat_idx}',
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)

    skipped = d_sae - n_active
    title = f'{n_active} SAE Feature Activation Heatmaps (d1 vs d2)'
    if skipped:
        title += f' — {skipped} dead latents hidden'
    fig.update_layout(
        title_text=title,
        height=figsize[1] * 100,  # Convert inches to pixels
        width=figsize[0] * 100,
        showlegend=False,
    )
    
    return fig


def create_firing_rate_histogram(sae_acts_all, figsize=(10, 6)):
    """
    Create seaborn histogram of feature firing rates.

    Args:
        sae_acts_all: SAE activations [n_samples, d_sae]
        figsize: Figure size

    Returns:
        fig: Matplotlib figure object
    """
    firing_rate = (sae_acts_all > 0).float().mean(dim=0).numpy()
    mean_rate = firing_rate.mean()

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(firing_rate, bins=50, ax=ax, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(mean_rate, color="crimson", linestyle="--", label=f"Mean: {mean_rate:.4f}")
    ax.set_xlabel("Firing Rate")
    ax.set_ylabel("Number of Features")
    ax.set_title("Distribution of Feature Firing Rates")
    ax.legend()
    sns.despine(ax=ax)

    return fig


def _compute_act_matrices(d1_all, d2_all, sae_acts_all, feature_indices, n_digits):
    """Build mean-activation matrices for given feature indices. Returns [n_feats, n_digits, n_digits]."""
    n_feats = len(feature_indices)
    act_matrices = torch.zeros(n_feats, n_digits, n_digits)
    count_matrix = torch.zeros(n_digits, n_digits)

    for i in range(len(d1_all)):
        d1, d2 = d1_all[i].item(), d2_all[i].item()
        count_matrix[d1, d2] += 1
        for j, feat_idx in enumerate(feature_indices):
            act_matrices[j, d1, d2] += sae_acts_all[i, feat_idx]

    act_matrices = act_matrices / count_matrix.clamp(min=1)
    return act_matrices


def create_feature_heatmaps_seaborn(d1_all, d2_all, sae_acts_all, feature_indices, n_digits=100, ncols=3, skip_dead_latents=True, shared_scale=False):
    """
    Create a compact seaborn grid of heatmaps for a specified subset of features.

    Args:
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: SAE activations [n_samples, d_sae]
        feature_indices: List of feature indices to plot
        n_digits: Number of possible digit values
        ncols: Number of columns in the grid
        skip_dead_latents: If True, omit features that never fire (default True)

    Returns:
        fig: Matplotlib figure object
    """
    if skip_dead_latents:
        feature_indices = [i for i in feature_indices if sae_acts_all[:, i].sum() != 0]
    n_feats = len(feature_indices)
    ncols = min(ncols, n_feats)
    nrows = int(np.ceil(n_feats / ncols))

    act_matrices = _compute_act_matrices(d1_all, d2_all, sae_acts_all, feature_indices, n_digits)

    global_max = float(max(m.max() for m in act_matrices)) if shared_scale else None

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for j, feat_idx in enumerate(feature_indices):
        ax = axes[j // ncols, j % ncols]
        mat = act_matrices[j].numpy()
        sns.heatmap(mat, ax=ax, cmap="viridis", xticklabels=False, yticklabels=False,
                    vmin=0 if shared_scale else None,
                    vmax=global_max if shared_scale else None,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(f"Latent {feat_idx}", fontsize=10)
        ax.set_xlabel("d2")
        ax.set_ylabel("d1")

    # Hide unused axes
    for j in range(n_feats, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    plt.tight_layout()
    return fig


def create_alpha_diff_correlation_plot(sae_acts_all, alpha_d1_all, alpha_d2_all, figsize=(12, 5)):
    """
    Two-panel seaborn figure: bar chart of per-feature alpha-diff correlations,
    and a scatter of the top correlated feature's activations vs alpha_diff.

    Args:
        sae_acts_all: SAE activations [n_samples, d_sae]
        alpha_d1_all: Attention weights SEP→d1 [n_samples]
        alpha_d2_all: Attention weights SEP→d2 [n_samples]
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
        correlations: np.ndarray of per-feature correlations
    """
    from .activation_collection import identify_special_features

    info = identify_special_features(sae_acts_all, alpha_d1_all, alpha_d2_all, threshold=0.0)
    correlations = info["all_correlations"]
    d_sae = len(correlations)
    alpha_diff = (alpha_d1_all - alpha_d2_all).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: bar chart of all correlations, coloured by sign
    colors = ["crimson" if c > 0 else "steelblue" for c in correlations]
    ax1.bar(np.arange(d_sae), correlations, color=colors, width=0.8)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(0.5, color="crimson", linestyle="--", linewidth=0.8, label="|r|=0.5")
    ax1.axhline(-0.5, color="crimson", linestyle="--", linewidth=0.8)
    ax1.set_yscale("symlog", linthresh=0.05)
    ax1.set_xlabel("Feature index")
    ax1.set_ylabel("Correlation with α_d1 − α_d2 (symlog)")
    ax1.set_title("Feature–alpha_diff correlations")
    ax1.legend(fontsize=8)
    sns.despine(ax=ax1)

    # Right: scatter for the most correlated feature
    top_feat = int(np.argmax(np.abs(correlations)))
    feat_acts = sae_acts_all[:, top_feat].numpy()
    active = feat_acts > 0
    ax2.scatter(alpha_diff[active], feat_acts[active], alpha=0.3, s=10, color="steelblue",
                label="active")
    ax2.scatter(alpha_diff[~active], feat_acts[~active], alpha=0.1, s=5, color="grey",
                label="inactive")
    ax2.set_xlabel("α_d1 − α_d2")
    ax2.set_ylabel(f"F{top_feat} activation")
    ax2.set_title(f"F{top_feat} vs alpha_diff  (r={correlations[top_feat]:.3f})")
    ax2.legend(fontsize=8)
    sns.despine(ax=ax2)

    plt.tight_layout()
    return fig, correlations
