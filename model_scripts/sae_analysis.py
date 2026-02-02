"""
SAE Analysis Utilities

Functions for analyzing and visualizing trained SAE features.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def collect_sae_activations(model, sae, val_dl, act_mean, layer_idx=0, sep_idx=2, device="cuda"):
    """
    Collect SAE latent activations and input pairs for all validation data.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        val_dl: Validation dataloader
        act_mean: Mean activation for centering
        layer_idx: Layer to extract activations from
        sep_idx: SEP token position
        device: Device to use
    
    Returns:
        d1_all: Tensor of d1 values
        d2_all: Tensor of d2 values
        sae_acts_all: Tensor of SAE latent activations [n_samples, d_sae]
    """
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    all_d1 = []
    all_d2 = []
    all_sae_acts = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(val_dl, desc="Collecting SAE activations", leave=False):
            inputs = inputs.to(device)
            
            # Extract d1, d2 values
            d1 = inputs[:, 0]
            d2 = inputs[:, 1]
            
            # Get SEP token activations
            _, cache = model.run_with_cache(
                inputs,
                stop_at_layer=layer_idx + 1,
                names_filter=[hook_name_resid]
            )
            sep_acts = cache[hook_name_resid][:, sep_idx, :]
            
            # Run SAE encoding
            sep_acts_centered = sep_acts - act_mean.to(sep_acts.device)
            sae_z = sae.encode(sep_acts_centered, use_threshold=True)
            
            # Store
            all_d1.append(d1.cpu())
            all_d2.append(d2.cpu())
            all_sae_acts.append(sae_z.cpu())
    
    d1_all = torch.cat(all_d1)
    d2_all = torch.cat(all_d2)
    sae_acts_all = torch.cat(all_sae_acts)
    
    return d1_all, d2_all, sae_acts_all


def create_feature_heatmaps(d1_all, d2_all, sae_acts_all, n_digits=100, figsize=(25, 25)):
    """
    Create grid of heatmaps showing activation patterns for all SAE features.
    
    Args:
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: Tensor of SAE activations [n_samples, d_sae]
        n_digits: Number of possible digit values
        figsize: Figure size for the plot
    
    Returns:
        fig: Matplotlib figure object
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
    
    # Create grid layout
    grid_size = int(np.ceil(np.sqrt(d_sae)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle(f'All {d_sae} SAE Feature Activation Heatmaps (d1 vs d2)', fontsize=16, y=0.995)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if d_sae > 1 else [axes]
    
    for feat_idx in range(d_sae):
        ax = axes_flat[feat_idx]
        
        im = ax.imshow(
            all_act_matrices[feat_idx].numpy(),
            cmap='viridis',
            origin='lower',
            aspect='auto'
        )
        ax.set_title(f'F{feat_idx}', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(d_sae, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    return fig


def compute_reconstruction_metrics(model, sae, val_dl, act_mean, layer_idx=0, sep_idx=2, device="cuda"):
    """
    Compute reconstruction quality metrics for the SAE.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        val_dl: Validation dataloader
        act_mean: Mean activation for centering
        layer_idx: Layer to extract activations from
        sep_idx: SEP token position
        device: Device to use
    
    Returns:
        dict with keys: mse, explained_variance, per_sample_mse
    """
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    all_mses = []
    all_orig_vars = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(val_dl, desc="Computing reconstruction metrics", leave=False):
            inputs = inputs.to(device)
            
            # Get original activations
            _, cache = model.run_with_cache(
                inputs,
                stop_at_layer=layer_idx + 1,
                names_filter=[hook_name_resid]
            )
            sep_acts = cache[hook_name_resid][:, sep_idx, :]
            
            # Encode and decode through SAE
            sep_acts_centered = sep_acts - act_mean.to(sep_acts.device)
            sae_z = sae.encode(sep_acts_centered, use_threshold=True)
            reconstructed = sae.decode(sae_z)
            
            # Compute MSE per sample
            mse = ((sep_acts_centered - reconstructed) ** 2).mean(dim=1)
            all_mses.append(mse.cpu())
            
            # Track variance for explained variance calculation
            all_orig_vars.append((sep_acts_centered ** 2).mean(dim=1).cpu())
    
    all_mses = torch.cat(all_mses)
    all_orig_vars = torch.cat(all_orig_vars)
    
    mean_mse = all_mses.mean().item()
    mean_var = all_orig_vars.mean().item()
    explained_variance = 1 - (mean_mse / mean_var) if mean_var > 0 else 0
    
    return {
        "mse": mean_mse,
        "explained_variance": explained_variance,
        "per_sample_mse": all_mses,
    }


def collect_attention_patterns(model, val_dl, layer_idx=0, sep_idx=2, device="cuda"):
    """
    Collect attention patterns from SEP token to d1 and d2.
    
    Args:
        model: Base transformer model
        val_dl: Validation dataloader
        layer_idx: Layer to extract attention from
        sep_idx: SEP token position
        device: Device to use
    
    Returns:
        alpha_d1_all: Attention from SEP to d1
        alpha_d2_all: Attention from SEP to d2
    """
    hook_name_attn = f"blocks.{layer_idx}.attn.hook_pattern"
    
    all_alpha_d1 = []
    all_alpha_d2 = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(val_dl, desc="Collecting attention patterns", leave=False):
            inputs = inputs.to(device)
            
            _, cache = model.run_with_cache(
                inputs,
                stop_at_layer=layer_idx + 1,
                names_filter=[hook_name_attn]
            )
            
            # Get attention pattern: [batch, n_heads, seq, seq]
            attn_pattern = cache[hook_name_attn][:, 0, :, :]  # [batch, seq, seq] (single head)
            alpha_d1 = attn_pattern[:, sep_idx, 0]  # SEP attending to d1
            alpha_d2 = attn_pattern[:, sep_idx, 1]  # SEP attending to d2
            
            all_alpha_d1.append(alpha_d1.cpu())
            all_alpha_d2.append(alpha_d2.cpu())
    
    return torch.cat(all_alpha_d1), torch.cat(all_alpha_d2)


def identify_special_features(sae_acts_all, alpha_d1_all, alpha_d2_all, threshold=0.5):
    """
    Identify 'special' features that correlate strongly with attention difference.
    
    Args:
        sae_acts_all: SAE activations [n_samples, d_sae]
        alpha_d1_all: Attention from SEP to d1 [n_samples]
        alpha_d2_all: Attention from SEP to d2 [n_samples]
        threshold: Correlation threshold for identifying special features
    
    Returns:
        dict with special feature statistics
    """
    d_sae = sae_acts_all.shape[1]
    alpha_diff = (alpha_d1_all - alpha_d2_all).numpy()
    
    special_features = []
    correlations = []
    
    for feat_idx in range(d_sae):
        feat_acts = sae_acts_all[:, feat_idx].numpy()
        
        # Skip if feature never fires
        if feat_acts.sum() == 0:
            correlations.append(0.0)
            continue
        
        corr = np.corrcoef(feat_acts, alpha_diff)[0, 1]
        correlations.append(corr)
        
        if abs(corr) > threshold:
            special_features.append({
                "feature_idx": feat_idx,
                "correlation": corr,
                "type": "d1_favoring" if corr > 0 else "d2_favoring",
            })
    
    correlations = np.array(correlations)
    
    return {
        "special_features": special_features,
        "n_special_features": len(special_features),
        "max_correlation": float(np.abs(correlations).max()),
        "mean_abs_correlation": float(np.abs(correlations).mean()),
        "all_correlations": correlations,
    }


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
