"""
SAE Analysis Utilities

Functions for analyzing and visualizing trained SAE features.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE


# ============================================================================
# Helper Functions (internal)
# ============================================================================

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


# ============================================================================
# Hook Creation Functions
# ============================================================================

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


# ============================================================================
# SAE Activation Collection & Analysis
# ============================================================================


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


def collect_attention_weights(model, dataloader, sep_idx, device="cuda"):
    """
    Collect attention weights from SEP token to input positions (d1 and d2).
    
    Args:
        model: Base transformer model
        dataloader: DataLoader with input data
        sep_idx: SEP token position
        device: Device to use
    
    Returns:
        alpha_d1_all: Tensor of attention weights from SEP to d1 [n_samples]
        alpha_d2_all: Tensor of attention weights from SEP to d2 [n_samples]
    """
    alpha_d1_all = []
    alpha_d2_all = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Extracting attention weights", leave=False):
            inputs = inputs.to(device)
            
            # Run model with attention cache
            logits, cache = model.run_with_cache(inputs)
            
            # Get attention scores from SEP token in layer 0
            attn_layer0 = cache["blocks.0.attn.hook_attn_scores"]  # [batch, n_heads, seq_len, seq_len]
            
            # SEP attends to d1 (position 0) and d2 (position 1)
            alpha_d1 = attn_layer0[:, :, sep_idx, 0].mean(dim=1)  # Average over heads
            alpha_d2 = attn_layer0[:, :, sep_idx, 1].mean(dim=1)
            
            alpha_d1_all.append(alpha_d1.cpu())
            alpha_d2_all.append(alpha_d2.cpu())
    
    alpha_d1_all = torch.cat(alpha_d1_all)
    alpha_d2_all = torch.cat(alpha_d2_all)
    
    return alpha_d1_all, alpha_d2_all


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


def compute_sae_reconstruction_accuracy(model, sae, val_dl, act_mean, layer_idx=0, sep_idx=2, device="cuda"):
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
            
            # Get attention pattern: [batch, n_heads, seq, seq]
            attn_pattern = _extract_activations(model, inputs, layer_idx, hook_name_attn)[:, 0, :, :]  # [batch, seq, seq] (single head)
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


# ============================================================================
# SAE Loading Functions
# ============================================================================

def load_sae_from_local(sae_name, d_model, device="cuda", sae_dir="../results/sae_models"):
    """
    Load a Sparse Autoencoder (SAE) from local checkpoint.
    
    Args:
        sae_name: Name of the SAE file (e.g., 'sae_d100_k4_50ksteps_2layer_100dig_64d.pt')
        d_model: Dimension of model activations
        device: Device to load SAE on
        sae_dir: Directory containing SAE checkpoints
    
    Returns:
        dict with keys:
            - sae: Loaded BatchTopKSAE model
            - act_mean: Activation mean for centering
            - config: SAE configuration dict
            - checkpoint: Full checkpoint dict
    """
    # Load checkpoint
    sae_path = os.path.join(sae_dir, sae_name)
    checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    
    # Extract config
    sae_cfg = checkpoint.get("cfg", {})
    d_sae = sae_cfg.get("dict_size", sae_cfg.get("d_sae", 256))
    top_k = sae_cfg.get("k", 4)
    
    # Create SAE instance
    sae = BatchTopKSAE(
        activation_dim=d_model,
        dict_size=d_sae,
        k=top_k
    ).to(device)
    
    # Load state dict (handle both old and new formats)
    state_dict = checkpoint["state_dict"]
    if "W_enc" in state_dict:
        # Legacy format conversion
        new_state_dict = {
            "encoder.weight": state_dict["W_enc"].T,
            "encoder.bias": state_dict["b_enc"],
            "decoder.weight": state_dict["W_dec"],
            "decoder.bias": state_dict["b_dec"],
        }
        sae.load_state_dict(new_state_dict)
    else:
        # New format
        sae.load_state_dict(state_dict)
    
    # Extract activation mean
    act_mean = checkpoint.get("act_mean", torch.zeros(d_model)).to(device)
    
    print(f"✓ Loaded SAE from {sae_path}")
    print(f"  - Dictionary size: {d_sae}")
    print(f"  - Top-K: {top_k}")
    if "final_loss" in checkpoint:
        print(f"  - Final loss: {checkpoint['final_loss']:.6f}")
    if "final_l0" in checkpoint:
        print(f"  - Final L0: {checkpoint['final_l0']:.2f}")
    
    return {
        "sae": sae,
        "act_mean": act_mean,
        "config": {
            "dict_size": d_sae,
            "d_sae": d_sae,
            "k": top_k,
            "top_k": top_k,
            "activation_dim": d_model,
            **sae_cfg
        },
        "checkpoint": checkpoint,
    }


def load_sae_from_wandb_run(run_id, project="theo-farrell99-durham-university/list-comp", 
                            download_dir="./wandb_downloads", device="cuda"):
    """
    Load an SAE model from a W&B run.
    
    Args:
        run_id: W&B run ID (e.g., "nqie9jok")
        project: W&B project path (default: "theo-farrell99-durham-university/list-comp")
        download_dir: Where to download artifacts (default: "./wandb_downloads")
        device: Device to load model on
    
    Returns:
        dict with keys:
            - sae: Loaded BatchTopKSAE model
            - act_mean: Activation mean for centering
            - config: SAE configuration
            - run_config: Full W&B run config
            - checkpoint: Full checkpoint dict
    """
    api = wandb.Api()
    
    # Get the run
    print(f"Fetching run {run_id}...")
    run = api.run(f"{project}/{run_id}")
    
    # Get run config
    run_config = run.config
    d_sae = run_config.get('d_sae')
    top_k = run_config.get('top_k')
    seed = run_config.get('seed')
    
    print(f"Run config: d_sae={d_sae}, k={top_k}, seed={seed}")
    
    # Find and download the SAE artifact
    artifact_name = f"sae-d{d_sae}-k{top_k}-seed{seed}"
    print(f"Downloading artifact: {artifact_name}")
    
    try:
        artifact = run.use_artifact(f"{artifact_name}:latest")
        artifact_dir = artifact.download(root=download_dir)
        
        # Find the .pt file in the artifact directory
        pt_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if not pt_files:
            raise FileNotFoundError(f"No .pt file found in artifact directory: {artifact_dir}")
        
        sae_path = os.path.join(artifact_dir, pt_files[0])
        print(f"Loading SAE from: {sae_path}")
        
    except Exception as e:
        print(f"Failed to download artifact: {e}")
        print("Trying local file path...")
        
        # Fallback to local file
        model_name = run_config.get('model_name', '2layer_100dig_64d')
        lr = run_config.get('lr')
        sae_filename = f"sae_d{d_sae}_k{top_k}_lr{lr}_seed{seed}_{model_name}.pt"
        sae_path = os.path.join('../results/sae_models/sweep_runs', sae_filename)
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE not found at: {sae_path}")
        
        print(f"Loading SAE from local: {sae_path}")
    
    # Load checkpoint
    checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    sae_config = checkpoint.get("cfg", {})
    
    # Extract config
    activation_dim = sae_config.get("activation_dim", sae_config.get("d_model", 64))
    dict_size = sae_config.get("dict_size", sae_config.get("d_sae", d_sae))
    k = sae_config.get("k", top_k)
    
    # Initialize SAE
    sae = BatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=dict_size,
        k=k
    ).to(device)
    
    # Load state dict
    sae.load_state_dict(checkpoint["state_dict"])
    act_mean = checkpoint["act_mean"].to(device)
    
    print(f"✓ Loaded SAE: d_sae={dict_size}, k={k}")
    print(f"  Final loss: {checkpoint.get('final_loss', 'N/A')}")
    print(f"  Final L0: {checkpoint.get('final_l0', 'N/A')}")
    
    return {
        "sae": sae,
        "act_mean": act_mean,
        "config": sae_config,
        "run_config": run_config,
        "checkpoint": checkpoint,
    }


def feature_steering_experiment(
    model, sae, act_mean, feature_idx,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=2, n_digits=100,
    scale_factors=None, n_test_cases=5, seed=42,
    device=None, plot=True, save_dir=None
):
    """
    Perform feature steering experiment by scaling a specific SAE feature's activation.
    
    Tests how scaling a feature's activation affects model outputs across different inputs.
    Samples test cases from inputs where the feature actually fires.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the SAE feature to steer
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: SAE activations [n_samples, d_sae]
        dataset: PyTorch dataset for getting inputs
        layer_idx: Layer to patch activations at
        sep_idx: SEP token position
        n_digits: Number of possible digit values
        scale_factors: Array of scale factors to test (default: [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        n_test_cases: Number of test cases to sample
        seed: Random seed for sampling
        device: Device to use (default: auto-detect from model)
        plot: Whether to create visualization
        save_dir: Directory to save plot (if None and plot=True, shows plot)
    
    Returns:
        all_results: List of dicts with keys:
            - 'd1', 'd2': Input pair
            - 'scales': Scale factors used
            - 'logit_d1_o1', 'logit_d2_o1': Logits at output position 1
            - 'logit_d1_o2', 'logit_d2_o2': Logits at output position 2
            - 'output_o1', 'output_o2': Predicted outputs
            - 'order_feat_orig': Original feature activation
    """
    if scale_factors is None:
        scale_factors = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Auto-detect device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Select test cases where the feature actually fires
    active_indices = torch.where(sae_acts_all[:, feature_idx] > 0)[0]
    print(f"Feature {feature_idx} fires on {len(active_indices)} / {len(d1_all)} inputs")
    
    # Sample from active inputs only
    np.random.seed(seed)
    n_samples = min(n_test_cases, len(active_indices))
    test_indices = np.random.choice(active_indices.numpy(), size=n_samples, replace=False)
    test_pairs = [(d1_all[i].item(), d2_all[i].item()) for i in test_indices]
    
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    # Storage for results
    all_results = []
    
    for d1_val, d2_val in test_pairs:
        # Find this pair in dataset
        mask = (d1_all == d1_val) & (d2_all == d2_val)
        if mask.sum() == 0:
            continue
        idx = torch.where(mask)[0][0].item()
        
        # Get inputs from dataset
        inputs_i = dataset[idx][0].unsqueeze(0).to(device)
        z_orig = sae_acts_all[idx].clone().to(device)
        feat_orig = z_orig[feature_idx].item()
        
        logit_d1_at_o1 = []
        logit_d2_at_o1 = []
        logit_d1_at_o2 = []
        logit_d2_at_o2 = []
        output_o1 = []
        output_o2 = []
        
        for scale in scale_factors:
            z_scaled = z_orig.clone()
            z_scaled[feature_idx] = feat_orig * scale
            
            recon = sae.decode(z_scaled.unsqueeze(0))
            
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    inputs_i,
                    fwd_hooks=[(hook_name_resid, make_sae_patch_hook(recon, act_mean, sep_idx))]
                )
            
            # Get logits at o1 (position -2) and o2 (position -1)
            logits_o1 = patched_logits[0, -2, :n_digits]
            logits_o2 = patched_logits[0, -1, :n_digits]
            
            logit_d1_at_o1.append(logits_o1[d1_val].item())
            logit_d2_at_o1.append(logits_o1[d2_val].item())
            logit_d1_at_o2.append(logits_o2[d1_val].item())
            logit_d2_at_o2.append(logits_o2[d2_val].item())
            output_o1.append(logits_o1.argmax().item())
            output_o2.append(logits_o2.argmax().item())
        
        all_results.append({
            'd1': d1_val, 'd2': d2_val,
            'scales': scale_factors,
            'logit_d1_o1': logit_d1_at_o1,
            'logit_d2_o1': logit_d2_at_o1,
            'logit_d1_o2': logit_d1_at_o2,
            'logit_d2_o2': logit_d2_at_o2,
            'output_o1': output_o1,
            'output_o2': output_o2,
            'order_feat_orig': feat_orig,
        })
    
    # Create visualization if requested
    if plot and len(all_results) > 0:
        fig, axes = plt.subplots(2, len(all_results), figsize=(4*len(all_results), 8), squeeze=False)
        
        for col, result in enumerate(all_results):
            d1, d2 = result['d1'], result['d2']
            scales = result['scales']
            
            # Top row: Logits at o1 position
            ax1 = axes[0, col]
            ax1.plot(scales, result['logit_d1_o1'], 'b-o', label=f'd1={d1} logit', markersize=4)
            ax1.plot(scales, result['logit_d2_o1'], 'r-s', label=f'd2={d2} logit', markersize=4)
            ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Original')
            ax1.axvline(x=0.0, color='red', linestyle=':', alpha=0.5)
            ax1.set_xlabel(f'Feature {feature_idx} Scale')
            ax1.set_ylabel('Logit at o1')
            ax1.set_title(f'Input ({d1}, {d2})\nf{feature_idx}_orig={result["order_feat_orig"]:.2f}')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Bottom row: Logits at o2 position
            ax2 = axes[1, col]
            ax2.plot(scales, result['logit_d1_o2'], 'b-o', label=f'd1={d1} logit', markersize=4)
            ax2.plot(scales, result['logit_d2_o2'], 'r-s', label=f'd2={d2} logit', markersize=4)
            ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Original')
            ax2.axvline(x=0.0, color='red', linestyle=':', alpha=0.5)
            ax2.set_xlabel(f'Feature {feature_idx} Scale')
            ax2.set_ylabel('Logit at o2')
            ax2.set_title(f'Logits at Output Position 2')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            import os
            save_path = os.path.join(save_dir, f'feature_{feature_idx}_logit_steering.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    return all_results


def compare_sweep_runs(project="theo-farrell99-durham-university/list-comp", 
                       sweep_id="wmhceuqf"):
    """
    Fetch summary statistics for all runs in a sweep.
    
    Args:
        project: W&B project path
        sweep_id: W&B sweep ID
    
    Returns:
        pandas DataFrame with run statistics
    """
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    
    runs_data = []
    for run in sweep.runs:
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "d_sae": run.config.get("d_sae"),
            "top_k": run.config.get("top_k"),
            "lr": run.config.get("lr"),
            "seed": run.config.get("seed"),
            "final_loss": run.summary.get("final_loss"),
            "final_l0": run.summary.get("final_l0"),
            "avg_l0": run.summary.get("avg_l0"),
            "dead_features": run.summary.get("dead_features"),
            "dead_features_pct": run.summary.get("dead_features_pct"),
            "reconstruction_mse": run.summary.get("reconstruction_mse"),
            "explained_variance": run.summary.get("explained_variance"),
            "n_special_features": run.summary.get("n_special_features"),
            "special_features_pct": run.summary.get("special_features_pct"),
            "max_attn_correlation": run.summary.get("max_attn_correlation"),
        }
        runs_data.append(run_data)
    
    df = pd.DataFrame(runs_data)
    return df.sort_values("explained_variance", ascending=False)
