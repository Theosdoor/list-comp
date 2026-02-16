"""
SAE Hook Utilities

Helper functions for creating hooks that patch SAE-reconstructed activations
into transformer models during forward passes.
"""

import torch


def _encode_through_sae(activations, sae, act_mean, decode=False, use_threshold=True):
    """
    Helper to encode (and optionally decode) activations through SAE.
    
    Args:
        activations: Input activations [batch, d_model]
        sae: Trained SAE model
        act_mean: Activation mean for centering
        decode: Whether to decode back to activation space
        use_threshold: Whether to use threshold in encoding
    
    Returns:
        If decode=True: reconstructed activations [batch, d_model]
        If decode=False: SAE latent codes [batch, d_sae]
    """
    activations_centered = activations - act_mean.to(activations.device)
    sae_z = sae.encode(activations_centered, use_threshold=use_threshold)
    
    if decode:
        reconstructed = sae.decode(sae_z)
        return reconstructed
    return sae_z


def _extract_activations(model, inputs, layer_idx, hook_name, stop_at_layer=None):
    """
    Helper to extract activations from model via cache.
    
    Args:
        model: Base transformer model
        inputs: Input tensor
        layer_idx: Layer index to extract from
        hook_name: Name of the hook to extract
        stop_at_layer: Layer to stop at (defaults to layer_idx + 1)
    
    Returns:
        Activations tensor from the specified hook
    """
    if stop_at_layer is None:
        stop_at_layer = layer_idx + 1
    
    _, cache = model.run_with_cache(
        inputs,
        stop_at_layer=stop_at_layer,
        names_filter=[hook_name]
    )
    return cache[hook_name]


def make_sae_patch_hook(reconstructed_acts, act_mean, sep_idx):
    """
    Create a hook that patches pre-computed SAE-reconstructed activations at the SEP token position.
    
    Args:
        reconstructed_acts: SAE-decoded activations (mean-centered)
        act_mean: Activation mean to add back (SAE outputs are mean-centered)
        sep_idx: SEP token position
    
    Returns:
        hook_fn: Hook function that can be used with model.run_with_hooks
    """
    def hook_fn(activations, hook):
        activations = activations.clone()
        # Add back the activation mean (SAE outputs are mean-centered)
        activations[:, sep_idx, :] = reconstructed_acts + act_mean.to(activations.device)
        return activations
    return hook_fn


def make_dynamic_sae_patch_hook(sae, act_mean, sep_idx):
    """
    Create a hook that dynamically encodes/decodes activations through SAE during forward pass.
    
    Args:
        sae: Trained SAE model
        act_mean: Activation mean for centering
        sep_idx: SEP token position
    
    Returns:
        hook_fn: Hook function that can be used with model.run_with_hooks
    """
    def hook_fn(activations, hook):
        activations = activations.clone()
        # Get SEP token activations and reconstruct through SAE
        sep_acts = activations[:, sep_idx, :]
        reconstructed = _encode_through_sae(sep_acts, sae, act_mean, decode=True)
        # Replace with reconstruction (add mean back)
        activations[:, sep_idx, :] = reconstructed + act_mean.to(reconstructed.device)
        return activations
    return hook_fn


def make_batched_sae_patch_hook(batch_recon, act_mean, sep_idx):
    """
    Create a hook that patches batched SAE-reconstructed activations at the SEP token position.
    
    This is useful when processing multiple inputs in parallel with different SAE reconstructions.
    
    Args:
        batch_recon: Batched SAE-decoded activations [batch_size, d_model] (mean-centered)
        act_mean: Activation mean to add back (SAE outputs are mean-centered)
        sep_idx: SEP token position
    
    Returns:
        hook_fn: Hook function that can be used with model.run_with_hooks
    """
    def hook_fn(activations, hook):
        activations = activations.clone()
        activations[:, sep_idx, :] = batch_recon + act_mean.to(activations.device)
        return activations
    return hook_fn
