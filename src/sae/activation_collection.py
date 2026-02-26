"""
SAE Activation Collection

Functions for collecting SAE activations and attention patterns from models.
"""

import torch
import numpy as np
from tqdm.auto import tqdm

from .hooks import _encode_through_sae, _extract_activations


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
            sep_acts = _extract_activations(model, inputs, layer_idx, hook_name_resid)[:, sep_idx, :]
            
            # Run SAE encoding
            sae_z = _encode_through_sae(sep_acts, sae, act_mean, decode=False)
            
            # Store
            all_d1.append(d1.cpu())
            all_d2.append(d2.cpu())
            all_sae_acts.append(sae_z.cpu())
    
    d1_all = torch.cat(all_d1)
    d2_all = torch.cat(all_d2)
    sae_acts_all = torch.cat(all_sae_acts)
    
    return d1_all, d2_all, sae_acts_all


def collect_attention_patterns(model, val_dl, layer_idx=0, sep_idx=2, device="cuda", use_scores=False):
    """
    Collect attention patterns from SEP token to d1 and d2.
    
    Args:
        model: Base transformer model
        val_dl: Validation dataloader
        layer_idx: Layer to extract attention from
        sep_idx: SEP token position
        device: Device to use
        use_scores: If True, use attention scores; if False, use attention patterns (default)
    
    Returns:
        alpha_d1_all: Attention from SEP to d1
        alpha_d2_all: Attention from SEP to d2
    """
    if use_scores:
        hook_name = f"blocks.{layer_idx}.attn.hook_attn_scores"
    else:
        hook_name = f"blocks.{layer_idx}.attn.hook_pattern"
    
    all_alpha_d1 = []
    all_alpha_d2 = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(val_dl, desc="Collecting attention patterns", leave=False):
            inputs = inputs.to(device)
            
            if use_scores:
                # Run model with cache and extract attention scores
                logits, cache = model.run_with_cache(inputs)
                attn = cache[hook_name]  # [batch, n_heads, seq_len, seq_len]
                # Average over heads
                alpha_d1 = attn[:, :, sep_idx, 0].mean(dim=1)
                alpha_d2 = attn[:, :, sep_idx, 1].mean(dim=1)
            else:
                # Get attention pattern: [batch, n_heads, seq, seq]
                attn_pattern = _extract_activations(model, inputs, layer_idx, hook_name)[:, 0, :, :]  # [batch, seq, seq] (single head)
                alpha_d1 = attn_pattern[:, sep_idx, 0]  # SEP attending to d1
                alpha_d2 = attn_pattern[:, sep_idx, 1]  # SEP attending to d2
            
            all_alpha_d1.append(alpha_d1.cpu())
            all_alpha_d2.append(alpha_d2.cpu())
    
    return torch.cat(all_alpha_d1), torch.cat(all_alpha_d2)


# Backward compatibility aliases
def collect_attention_weights(model, dataloader, sep_idx, device="cuda"):
    """
    Collect attention weights from SEP token to input positions (d1 and d2).
    
    DEPRECATED: Use collect_attention_patterns(use_scores=True) instead.
    
    Args:
        model: Base transformer model
        dataloader: DataLoader with input data
        sep_idx: SEP token position
        device: Device to use
    
    Returns:
        alpha_d1_all: Tensor of attention weights from SEP to d1 [n_samples]
        alpha_d2_all: Tensor of attention weights from SEP to d2 [n_samples]
    """
    return collect_attention_patterns(model, dataloader, layer_idx=0, sep_idx=sep_idx, 
                                     device=device, use_scores=True)


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
        
        # Compute correlation, handling NaN cases
        try:
            corr_matrix = np.corrcoef(feat_acts, alpha_diff)
            corr = corr_matrix[0, 1]
            # Replace NaN with 0 (happens when feature has 0 variance)
            if np.isnan(corr):
                corr = 0.0
        except (ValueError, RuntimeWarning):
            corr = 0.0
        
        correlations.append(corr)
        
        if abs(corr) > threshold:
            special_features.append({
                "feature_idx": feat_idx,
                "correlation": corr,
                "type": "d1_favoring" if corr > 0 else "d2_favoring",
            })
    
    correlations = np.array(correlations)
    # Replace NaNs in correlations array
    correlations = np.nan_to_num(correlations, nan=0.0)
    
    # Handle case where all correlations are 0
    max_corr = float(np.abs(correlations).max()) if len(correlations) > 0 else 0.0
    mean_corr = float(np.abs(correlations).mean()) if len(correlations) > 0 else 0.0
    
    return {
        "special_features": special_features,
        "n_special_features": len(special_features),
        "max_correlation": max_corr,
        "mean_abs_correlation": mean_corr,
        "all_correlations": correlations,
    }
