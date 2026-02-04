"""
SAE Analysis Utilities

Functions for analyzing and visualizing trained SAE features.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE


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
    
    def reconstruction_hook(activations, hook):
        """Replace activations with SAE reconstruction at SEP position."""
        # Get SEP token activations
        sep_acts = activations[:, sep_idx, :]
        # Center and reconstruct
        sep_acts_centered = sep_acts - act_mean.to(sep_acts.device)
        sae_z = sae.encode(sep_acts_centered, use_threshold=True)
        reconstructed = sae.decode(sae_z)
        # Replace with reconstruction (add mean back)
        activations[:, sep_idx, :] = reconstructed + act_mean.to(reconstructed.device)
        return activations
    
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


# NOTE - USE from src.utils.nb_utils import load_sae for local loading

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
