"""
SAE Metrics

Functions for computing SAE reconstruction quality metrics.
"""

import torch
from tqdm.auto import tqdm

from .hooks import _encode_through_sae, _extract_activations, make_dynamic_sae_patch_hook


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
            sep_acts = _extract_activations(model, inputs, layer_idx, hook_name_resid)[:, sep_idx, :]
            
            # Encode and decode through SAE
            sep_acts_centered = sep_acts - act_mean.to(sep_acts.device)
            reconstructed = _encode_through_sae(sep_acts, sae, act_mean, decode=True)
            
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


def compute_sae_patched_accuracy(model, sae, val_dl, act_mean, layer_idx=0, sep_idx=2, device="cuda"):
    """
    Compute model accuracy when using SAE-reconstructed activations instead of original.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        val_dl: Validation dataloader
        act_mean: Mean activation for centering
        layer_idx: Layer to extract activations from
        sep_idx: SEP token position (list_len)
        device: Device to use
    
    Returns:
        dict with keys: baseline_acc, reconstruction_acc, accuracy_drop, total_samples
    """
    from ..utils.runtime import _RUNTIME
    list_len = _RUNTIME.list_len
    
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    # 1. Baseline accuracy (original model)
    correct_baseline = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing baseline accuracy", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)[:, list_len + 1:]  # Only output positions
            preds = logits.argmax(dim=-1)
            correct_baseline += (preds == targets[:, list_len + 1:]).sum().item()
            total += preds.numel()
    
    baseline_acc = correct_baseline / total
    
    # 2. Accuracy with SAE reconstruction
    correct_recon = 0
    reconstruction_hook = make_dynamic_sae_patch_hook(sae, act_mean, sep_idx)
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing SAE reconstruction accuracy", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Run with reconstruction hook
            logits = model.run_with_hooks(
                inputs,
                fwd_hooks=[(hook_name_resid, reconstruction_hook)]
            )[:, list_len + 1:]  # Only output positions
            preds = logits.argmax(dim=-1)
            correct_recon += (preds == targets[:, list_len + 1:]).sum().item()
    
    recon_acc = correct_recon / total
    acc_drop = baseline_acc - recon_acc
    
    return {
        'baseline_acc': baseline_acc,
        'reconstruction_acc': recon_acc,
        'accuracy_drop': acc_drop,
        'total_samples': total
    }
