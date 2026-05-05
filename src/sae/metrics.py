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
    Compute downstream metrics for an SAE in three forward passes (baseline + patched + zero-ablation).

    Returns accuracy and CE loss together to avoid redundant passes.

    Returns:
        dict with keys:
            baseline_acc, reconstruction_acc, accuracy_drop, total_samples,
            baseline_ce, patched_ce, ce_increase, total_tokens,
            zero_ce, loss_recovered
    """
    from ..utils.runtime import _RUNTIME
    list_len = _RUNTIME.list_len

    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    reconstruction_hook = make_dynamic_sae_patch_hook(sae, act_mean, sep_idx)

    # Zero-ablation hook: replace SEP token residual with zeros
    def zero_ablation_hook(activations, hook):
        activations = activations.clone()
        activations[:, sep_idx, :] = 0.0
        return activations

    baseline_ce_total = 0.0
    patched_ce_total = 0.0
    zero_ce_total = 0.0
    correct_baseline = 0
    correct_patched = 0
    total_tokens = 0
    v = None

    # Baseline pass
    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing baseline metrics", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out_targets = targets[:, list_len + 1:]  # [batch, list_len]
            b, t = out_targets.shape

            baseline_logits = model(inputs)[:, list_len + 1:]  # [batch, list_len, vocab]
            if v is None:
                v = baseline_logits.shape[-1]
            baseline_ce_total += F.cross_entropy(
                baseline_logits.reshape(b * t, v),
                out_targets.reshape(b * t),
                reduction="sum",
            ).item()
            correct_baseline += (baseline_logits.argmax(dim=-1) == out_targets).sum().item()
            total_tokens += b * t

    # Patched pass
    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing patched metrics", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out_targets = targets[:, list_len + 1:]  # [batch, list_len]
            b, t = out_targets.shape

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

    # Zero-ablation pass
    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="Computing zero-ablation metrics", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out_targets = targets[:, list_len + 1:]  # [batch, list_len]
            b, t = out_targets.shape

            zero_logits = model.run_with_hooks(
                inputs,
                fwd_hooks=[(hook_name_resid, zero_ablation_hook)]
            )[:, list_len + 1:]
            zero_ce_total += F.cross_entropy(
                zero_logits.reshape(b * t, v),
                out_targets.reshape(b * t),
                reduction="sum",
            ).item()

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
        "baseline_ce": baseline_ce,
        "patched_ce": patched_ce,
        "ce_increase": patched_ce - baseline_ce,
        "zero_ce": zero_ce,
        "loss_recovered": loss_recovered,
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


# loss recoverred metric implementation
# from https://github.com/adamkarvonen/SAEBench/blob/main/sae_bench/evals/core/main.py

# @torch.no_grad()
# def get_recons_loss(
#     sae: SAE,
#     model: HookedRootModule,
#     batch_tokens: torch.Tensor,
#     activation_store: ActivationsStore,
#     compute_kl: bool,
#     compute_ce_loss: bool,
#     ignore_tokens: set[int | None] = set(),
#     exclude_special_tokens_from_reconstruction: bool = False,
#     model_kwargs: Mapping[str, Any] = {},
# ) -> dict[str, Any]:
#     hook_name = sae.cfg.hook_name
#     head_index = sae.cfg.hook_head_index

#     original_logits, original_ce_loss = model(
#         batch_tokens, return_type="both", loss_per_token=True, **model_kwargs
#     )

#     if len(ignore_tokens) > 0 and exclude_special_tokens_from_reconstruction:
#         mask = torch.logical_not(
#             torch.any(
#                 torch.stack([batch_tokens == token for token in ignore_tokens], dim=0),
#                 dim=0,
#             )
#         )
#     else:
#         mask = torch.ones_like(batch_tokens, dtype=torch.bool)

#     metrics = {}

#     # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
#     def standard_replacement_hook(activations: torch.Tensor, hook: Any):
#         original_device = activations.device
#         activations = activations.to(sae.device)

#         # SAE class agnost forward forward pass.
#         reconstructed_activations = sae.decode(sae.encode(activations)).to(
#             activations.dtype
#         )

#         reconstructed_activations = torch.where(
#             mask[..., None], reconstructed_activations, activations
#         )

#         return reconstructed_activations.to(original_device)

#     def all_head_replacement_hook(activations: torch.Tensor, hook: Any):
#         original_device = activations.device
#         activations = activations.to(sae.device)

#         # SAE class agnost forward forward pass.
#         new_activations = sae.decode(sae.encode(activations.flatten(-2, -1))).to(
#             activations.dtype
#         )

#         new_activations = new_activations.reshape(
#             activations.shape
#         )  # reshape to match original shape

#         # Apply mask to keep original activations for ignored tokens
#         new_activations = torch.where(
#             mask[..., None, None], new_activations, activations
#         )

#         return new_activations.to(original_device)

#     def single_head_replacement_hook(activations: torch.Tensor, hook: Any):
#         original_device = activations.device
#         activations = activations.to(sae.device)

#         # Create a copy of activations to modify
#         new_activations = activations.clone()

#         # Only reconstruct the specified head
#         head_activations = sae.decode(sae.encode(activations[:, :, head_index])).to(
#             activations.dtype
#         )

#         # Apply mask only to the reconstructed head
#         masked_head_activations = torch.where(
#             mask[..., None], head_activations, activations[:, :, head_index]
#         )
#         new_activations[:, :, head_index] = masked_head_activations

#         return new_activations.to(original_device)

#     def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):
#         original_device = activations.device
#         activations = activations.to(sae.device)
#         activations = torch.zeros_like(activations)
#         return activations.to(original_device)

#     def single_head_zero_ablate_hook(activations: torch.Tensor, hook: Any):
#         original_device = activations.device
#         activations = activations.to(sae.device)
#         activations[:, :, head_index] = torch.zeros_like(activations[:, :, head_index])
#         return activations.to(original_device)

#     # we would include hook z, except that we now have base SAE's
#     # which will do their own reshaping for hook z.
#     has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
#     if any(substring in hook_name for substring in has_head_dim_key_substrings):
#         if head_index is None:
#             replacement_hook = all_head_replacement_hook
#             zero_ablate_hook = standard_zero_ablate_hook
#         else:
#             replacement_hook = single_head_replacement_hook
#             zero_ablate_hook = single_head_zero_ablate_hook
#     else:
#         replacement_hook = standard_replacement_hook
#         zero_ablate_hook = standard_zero_ablate_hook

#     recons_logits, recons_ce_loss = model.run_with_hooks(
#         batch_tokens,
#         return_type="both",
#         fwd_hooks=[(hook_name, partial(replacement_hook))],
#         loss_per_token=True,
#         **model_kwargs,
#     )

#     zero_abl_logits, zero_abl_ce_loss = model.run_with_hooks(
#         batch_tokens,
#         return_type="both",
#         fwd_hooks=[(hook_name, zero_ablate_hook)],
#         loss_per_token=True,
#         **model_kwargs,
#     )

#     def kl(original_logits: torch.Tensor, new_logits: torch.Tensor):
#         original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
#         log_original_probs = torch.log(original_probs)
#         new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
#         log_new_probs = torch.log(new_probs)
#         kl_div = original_probs * (log_original_probs - log_new_probs)
#         kl_div = kl_div.sum(dim=-1)
#         return kl_div

#     if compute_kl:
#         recons_kl_div = kl(original_logits, recons_logits)
#         zero_abl_kl_div = kl(original_logits, zero_abl_logits)
#         metrics["kl_div_with_sae"] = recons_kl_div
#         metrics["kl_div_with_ablation"] = zero_abl_kl_div

#     if compute_ce_loss:
#         metrics["ce_loss_with_sae"] = recons_ce_loss
#         metrics["ce_loss_without_sae"] = original_ce_loss
#         metrics["ce_loss_with_ablation"] = zero_abl_ce_loss

#     return metrics
