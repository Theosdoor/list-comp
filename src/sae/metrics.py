"""
SAE Metrics

Functions for computing SAE reconstruction quality metrics.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .hooks import _encode_through_sae, _extract_activations, make_dynamic_sae_patch_hook, make_zero_sep_hook


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


def _output_targets(targets, list_len):
    return targets[:, list_len + 1:]


def _accumulate_output_ce(logits, out_targets):
    b, t = out_targets.shape
    v = logits.shape[-1]
    ce_total = F.cross_entropy(
        logits.reshape(b * t, v),
        out_targets.reshape(b * t),
        reduction="sum",
    ).item()
    correct = (logits.argmax(dim=-1) == out_targets).sum().item()
    return ce_total, correct, b * t


def _forward_output_logits(model, inputs, list_len, hook_name=None, hook_fn=None):
    if hook_name is None:
        logits = model(inputs)
    else:
        logits = model.run_with_hooks(inputs, fwd_hooks=[(hook_name, hook_fn)])
    return logits[:, list_len + 1:]


def compute_sae_downstream_metrics(model, sae, val_dl, act_mean, layer_idx=0, sep_idx=2, device="cuda"):
    """
    Compute downstream metrics for an SAE in three forward passes (baseline + patched + zero-ablation).

    Returns accuracy and loss_recovered metric together.

    Returns:
        dict with keys:
            baseline_acc, reconstruction_acc, accuracy_drop, total_samples,
            total_tokens, h_orig, h_star, h0, zero_ce, loss_recovered
    """
    from ..utils.runtime import _RUNTIME
    list_len = _RUNTIME.list_len

    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    reconstruction_hook = make_dynamic_sae_patch_hook(sae, act_mean, sep_idx)
    zero_ablation_hook = make_zero_sep_hook(sep_idx)

    baseline_ce_total = 0.0
    patched_ce_total = 0.0
    zero_ce_total = 0.0
    correct_baseline = 0
    correct_patched = 0
    total_tokens = 0

    # Extract vocab size upfront to avoid uninitialized use in later loops
    v = None
    with torch.no_grad():
        for inputs, targets in val_dl:
            baseline_logits = _forward_output_logits(model, inputs, list_len)
            v = baseline_logits.shape[-1]
            break
    
    if v is None:
        raise ValueError("Failed to determine vocab size from validation dataloader")

    # Baseline pass
    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing baseline metrics", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out_targets = _output_targets(targets, list_len)

            baseline_logits = _forward_output_logits(model, inputs, list_len)
            ce_total, correct, token_count = _accumulate_output_ce(baseline_logits, out_targets)
            baseline_ce_total += ce_total
            correct_baseline += correct
            total_tokens += token_count

    # Patched pass
    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing patched metrics", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out_targets = _output_targets(targets, list_len)

            patched_logits = _forward_output_logits(model, inputs, list_len, hook_name_resid, reconstruction_hook)
            ce_total, correct, _ = _accumulate_output_ce(patched_logits, out_targets)
            patched_ce_total += ce_total
            correct_patched += correct

    # Zero-ablation pass
    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing zero-ablation metrics", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out_targets = _output_targets(targets, list_len)

            zero_logits = _forward_output_logits(model, inputs, list_len, hook_name_resid, zero_ablation_hook)
            ce_total, _, _ = _accumulate_output_ce(zero_logits, out_targets)
            zero_ce_total += ce_total

    if total_tokens == 0:
        raise ValueError("Empty validation dataloader provided to compute_sae_downstream_metrics")

    baseline_ce = baseline_ce_total / total_tokens
    patched_ce = patched_ce_total / total_tokens
    zero_ce = zero_ce_total / total_tokens
    baseline_acc = correct_baseline / total_tokens
    patched_acc = correct_patched / total_tokens

    # Compute loss recovered
    denom = baseline_ce - zero_ce
    loss_recovered = (patched_ce - zero_ce) / denom if abs(denom) > 1e-10 else None

    return {
        "baseline_acc": baseline_acc,
        "reconstruction_acc": patched_acc,
        "accuracy_drop": baseline_acc - patched_acc,
        "total_samples": total_tokens,
        "total_tokens": total_tokens,
        "h_orig": baseline_ce,
        "h_star": patched_ce,
        "h0": zero_ce,
        "zero_ce": zero_ce,   # backward-compat alias for h0
        "loss_recovered": loss_recovered,
    }
