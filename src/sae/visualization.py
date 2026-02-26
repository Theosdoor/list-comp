"""
SAE Visualization Utilities

Functions for creating plots and visualizations of SAE features and activations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_feature_heatmaps(d1_all, d2_all, sae_acts_all, n_digits=100, figsize=(25, 25)):
    """
    Create interactive grid of heatmaps showing activation patterns for all SAE features.
    
    Args:
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: Tensor of SAE activations [n_samples, d_sae]
        n_digits: Number of possible digit values
        figsize: Figure size (width, height) in inches
    
    Returns:
        fig: Plotly Figure object (interactive)
    """
    d_sae = sae_acts_all.shape[1]
    n_samples = len(d1_all)
    
    # Compute all activation matrices
    all_act_matrices = torch.zeros(d_sae, n_digits, n_digits)
    count_matrix = torch.zeros(n_digits, n_digits)
    
    for i in range(n_samples):
        d1, d2 = d1_all[i].item(), d2_all[i].item()
        all_act_matrices[:, d1, d2] += sae_acts_all[i]
        count_matrix[d1, d2] += 1
    
    all_act_matrices = all_act_matrices / count_matrix.clamp(min=1)
    
    # Create subplot grid
    grid_size = int(np.ceil(np.sqrt(d_sae)))
    
    # Create subplot specs and titles
    subplot_titles = [f'F{i}' for i in range(d_sae)]
    # Add empty titles for unused subplots
    total_subplots = grid_size * grid_size
    subplot_titles.extend([''] * (total_subplots - d_sae))
    
    fig = make_subplots(
        rows=grid_size,
        cols=grid_size,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )
    
    # Add heatmaps
    for feat_idx in range(d_sae):
        row = feat_idx // grid_size + 1
        col = feat_idx % grid_size + 1
        
        fig.add_trace(
            go.Heatmap(
                z=all_act_matrices[feat_idx].numpy(),
                colorscale='Viridis',
                showscale=(feat_idx == d_sae - 1),  # Show colorbar only on last subplot
                hovertemplate='d1: %{x}<br>d2: %{y}<br>Activation: %{z:.4f}<extra></extra>',
                name=f'F{feat_idx}',
            ),
            row=row,
            col=col
        )
        
        # Update axes for this subplot
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)
    
    fig.update_layout(
        title_text=f'All {d_sae} SAE Feature Activation Heatmaps (d1 vs d2)',
        height=figsize[1] * 100,  # Convert inches to pixels
        width=figsize[0] * 100,
        showlegend=False,
    )
    
    return fig


def create_firing_rate_histogram(sae_acts_all, figsize=(10, 6)):
    """
    Create histogram of feature firing rates.
    
    Args:
        sae_acts_all: SAE activations [n_samples, d_sae]
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure object
    """
    firing_rate = (sae_acts_all > 0).float().mean(dim=0).numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(firing_rate, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Firing Rate')
    ax.set_ylabel('Number of Features')
    ax.set_title('Distribution of Feature Firing Rates')
    ax.axvline(firing_rate.mean(), color='red', linestyle='--', 
               label=f'Mean: {firing_rate.mean():.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
