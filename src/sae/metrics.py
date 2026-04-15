"""
SAE Metrics

Functions for computing SAE reconstruction quality metrics.
"""

import torch
import torch.nn.functional as F
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


def compute_sae_downstream_metrics(model, sae, val_dl, act_mean, layer_idx=0, sep_idx=2, device="cuda"):
    """
    Compute downstream metrics for an SAE in two forward passes (baseline + patched).

    Returns accuracy and CE loss together to avoid redundant passes.

    Returns:
        dict with keys:
            baseline_acc, reconstruction_acc, accuracy_drop, total_samples,
            baseline_ce, patched_ce, ce_increase, total_tokens
    """
    from ..utils.runtime import _RUNTIME
    list_len = _RUNTIME.list_len

    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    reconstruction_hook = make_dynamic_sae_patch_hook(sae, act_mean, sep_idx)

    baseline_ce_total = 0.0
    patched_ce_total = 0.0
    correct_baseline = 0
    correct_patched = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing SAE downstream metrics", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out_targets = targets[:, list_len + 1:]  # [batch, list_len]
            b, t = out_targets.shape

            # Baseline forward pass
            baseline_logits = model(inputs)[:, list_len + 1:]  # [batch, list_len, vocab]
            v = baseline_logits.shape[-1]
            baseline_ce_total += F.cross_entropy(
                baseline_logits.reshape(b * t, v),
                out_targets.reshape(b * t),
                reduction="sum",
            ).item()
            correct_baseline += (baseline_logits.argmax(dim=-1) == out_targets).sum().item()

            # Patched forward pass
            patched_logits = model.run_with_hooks(
                inputs,
                fwd_hooks=[(hook_name_resid, reconstruction_hook)]
            )[:, list_len + 1:]
            patched_ce_total += F.cross_entropy(
                patched_logits.reshape(b * t, v),
                out_targets.reshape(b * t),
                reduction="sum",
            ).item()
            correct_patched += (patched_logits.argmax(dim=-1) == out_targets).sum().item()

            total_tokens += b * t

    baseline_ce = baseline_ce_total / total_tokens
    patched_ce = patched_ce_total / total_tokens
    baseline_acc = correct_baseline / total_tokens
    patched_acc = correct_patched / total_tokens

    return {
        "baseline_acc": baseline_acc,
        "reconstruction_acc": patched_acc,
        "accuracy_drop": baseline_acc - patched_acc,
        "total_samples": total_tokens,
        "baseline_ce": baseline_ce,
        "patched_ce": patched_ce,
        "ce_increase": patched_ce - baseline_ce,
        "total_tokens": total_tokens,
    }


def compute_sae_patched_accuracy(model, sae, val_dl, act_mean, layer_idx=0, sep_idx=2, device="cuda"):
    """Thin wrapper around compute_sae_downstream_metrics for backward compatibility."""
    result = compute_sae_downstream_metrics(model, sae, val_dl, act_mean, layer_idx, sep_idx, device)
    return {k: result[k] for k in ("baseline_acc", "reconstruction_acc", "accuracy_drop", "total_samples")}


def compute_sae_patched_ce_loss(model, sae, val_dl, act_mean, layer_idx=0, sep_idx=2, device="cuda"):
    """Thin wrapper around compute_sae_downstream_metrics for backward compatibility."""
    result = compute_sae_downstream_metrics(model, sae, val_dl, act_mean, layer_idx, sep_idx, device)
    return {k: result[k] for k in ("baseline_ce", "patched_ce", "ce_increase", "total_tokens")}
