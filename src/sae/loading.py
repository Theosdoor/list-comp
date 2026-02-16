"""
SAE Loading Utilities

Functions for loading SAE models from local checkpoints or Weights & Biases.
"""

import os
import torch
import pandas as pd
import wandb

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE


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
