"""
SAE Loading Utilities

Functions for loading SAE models from local checkpoints or Weights & Biases.
"""

import os
import re
import torch
import pandas as pd
import wandb
from pathlib import Path
from collections import defaultdict

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from dictionary_learning.dictionary import JumpReluAutoEncoder
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE


def instantiate_sae_from_cfg(cfg: dict, d_model: int, device: str):
    """
    Single dispatch point: create an SAE instance from a checkpoint cfg dict.

    Supports sae_type in {"btk", "jumprelu", "matryoshka"}.
    Defaults to "btk" for checkpoints that predate the sae_type field.

    Args:
        cfg:     Checkpoint cfg dict (from checkpoint["cfg"]).
        d_model: Activation dimension (used when not present in cfg).
        device:  Device string.

    Returns:
        Instantiated SAE (not yet populated with weights — call load_state_dict separately).
    """
    sae_type = cfg.get("sae_type", "btk")
    activation_dim = cfg.get("activation_dim", cfg.get("d_model", d_model))
    dict_size = cfg.get("dict_size", cfg.get("d_sae", 256))

    if sae_type == "btk":
        k = cfg.get("k", 4)
        return BatchTopKSAE(activation_dim=activation_dim, dict_size=dict_size, k=k).to(device)

    elif sae_type == "jumprelu":
        return JumpReluAutoEncoder(activation_dim=activation_dim, dict_size=dict_size, device=device)

    elif sae_type == "matryoshka":
        k = cfg.get("k", 4)
        group_sizes = cfg.get("group_sizes")
        if group_sizes is None:
            n_groups = cfg.get("n_groups", 4)
            base = dict_size // n_groups
            group_sizes = [base] * (n_groups - 1) + [dict_size - base * (n_groups - 1)]
        return MatryoshkaBatchTopKSAE(
            activation_dim=activation_dim, dict_size=dict_size, k=k, group_sizes=group_sizes
        ).to(device)

    else:
        raise ValueError(
            f"Unknown sae_type '{sae_type}'. "
            f"Supported types: 'btk', 'jumprelu', 'matryoshka'. "
            f"To add a new type, register it in instantiate_sae_from_cfg()."
        )


def normalize_sae_state_dict(cfg: dict, state_dict: dict) -> dict:
    """Normalize checkpoint state dicts to the instantiated SAE module format."""
    sae_type = cfg.get("sae_type", "btk")
    if sae_type == "btk" and "W_enc" in state_dict:
        return {
            "encoder.weight": state_dict["W_enc"].T,
            "encoder.bias": state_dict["b_enc"],
            "decoder.weight": state_dict["W_dec"],
            "decoder.bias": state_dict["b_dec"],
        }
    return state_dict


def _activation_dim_from_checkpoint(cfg: dict, d_model: int | None, checkpoint: dict) -> int:
    """Infer the activation dimension without relying on filename conventions."""
    if "activation_dim" in cfg:
        return cfg["activation_dim"]
    if "d_model" in cfg:
        return cfg["d_model"]
    if d_model is not None:
        return d_model

    act_mean = checkpoint.get("act_mean")
    if act_mean is not None:
        return act_mean.numel()

    state_dict = checkpoint["state_dict"]
    if "W_enc" in state_dict:
        return state_dict["W_enc"].shape[0]
    if "encoder.weight" in state_dict:
        return state_dict["encoder.weight"].shape[1]
    if "W_dec" in state_dict:
        return state_dict["W_dec"].shape[1]
    if "decoder.weight" in state_dict:
        return state_dict["decoder.weight"].shape[1]

    raise ValueError("Could not infer SAE activation dimension; pass d_model explicitly")


def load_sae_from_checkpoint(checkpoint: dict, d_model: int | None, device: str):
    """
    Instantiate and populate an SAE from an already-loaded checkpoint dict.

    This is the canonical path for checkpoint config handling, legacy state-dict
    normalization, act_mean placement, and returned config assembly.
    """
    sae_cfg = checkpoint.get("cfg", {})
    activation_dim = _activation_dim_from_checkpoint(sae_cfg, d_model, checkpoint)
    d_sae = sae_cfg.get("dict_size", sae_cfg.get("d_sae", 256))

    sae = instantiate_sae_from_cfg(sae_cfg, activation_dim, device)
    state_dict = normalize_sae_state_dict(sae_cfg, checkpoint["state_dict"])
    sae.load_state_dict(state_dict)

    act_mean = checkpoint.get("act_mean", torch.zeros(activation_dim)).to(device)
    config = {
        "dict_size": d_sae,
        "d_sae": d_sae,
        "activation_dim": activation_dim,
        "act_mean": act_mean,
        **sae_cfg,
    }

    return {
        "sae": sae,
        "act_mean": act_mean,
        "config": config,
        "checkpoint": checkpoint,
    }


def load_sae_checkpoint(sae_path, d_model: int | None = None, device: str = "cuda"):
    """
    Load, instantiate, normalize, and populate an SAE checkpoint from disk.

    Returns a dict with keys: sae, act_mean, config, checkpoint.
    """
    sae_path = Path(sae_path)
    if sae_path.suffix != ".pt":
        raise ValueError(f"Expected a .pt file, got: {sae_path}")
    if not sae_path.exists():
        raise FileNotFoundError(f"No SAE checkpoint found at {sae_path}")

    checkpoint = torch.load(str(sae_path), map_location=device, weights_only=False)
    return load_sae_from_checkpoint(checkpoint, d_model, device)


def load_sae_from_local(sae_name, d_model, device="cuda", sae_dir="../sae_checkpoints"):
    """
    Load a Sparse Autoencoder (SAE) from local checkpoint.

    Args:
        sae_name: Name of the SAE file (e.g., 'btk_sae_d256_k3_lr1e-4_seed0_2layer_100dig_64d.pt')
        d_model: Dimension of model activations
        device: Device to load SAE on
        sae_dir: Directory containing SAE checkpoints

    Returns:
        dict with keys:
            - sae: Loaded SAE model
            - act_mean: Activation mean for centering
            - config: SAE configuration dict
            - checkpoint: Full checkpoint dict
    """
    sae_path = os.path.join(sae_dir, sae_name)
    result = load_sae_checkpoint(sae_path, d_model=d_model, device=device)
    checkpoint = result["checkpoint"]
    sae_cfg = checkpoint.get("cfg", {})
    d_sae = result["config"]["d_sae"]
    sae_type = sae_cfg.get("sae_type", "btk")

    print(f"✓ Loaded {sae_type} SAE from {sae_path}")
    print(f"  - Dictionary size: {d_sae}")
    if "final_loss" in checkpoint:
        print(f"  - Final loss: {checkpoint['final_loss']:.6f}")
    if "final_l0" in checkpoint:
        print(f"  - Final L0: {checkpoint['final_l0']:.2f}")

    return result


def load_sae_from_wandb_run(run_id, project=None,
                            download_dir="./wandb_downloads", device="cuda"):
    """
    Load an SAE model from a W&B run.
    
    Args:
        run_id: W&B run ID (e.g., "nqie9jok")
        project: W&B project path formatted as "entity/project".
        download_dir: Where to download artifacts (default: "./wandb_downloads")
        device: Device to load model on
    
    Returns:
        dict with keys:
            - sae: Loaded SAE model
            - act_mean: Activation mean for centering
            - config: SAE configuration
            - run_config: Full W&B run config
            - checkpoint: Full checkpoint dict
    """
    if project is None:
        raise ValueError("project must be provided as 'entity/project'")

    api = wandb.Api()
    
    # Get the run
    print(f"Fetching run {run_id}...")
    run = api.run(f"{project}/{run_id}")
    
    # Get run config
    run_config = run.config
    sae_type = run_config.get('sae_type', 'btk')
    d_sae = run_config.get('d_sae')
    top_k = run_config.get('top_k')
    seed = run_config.get('seed')

    print(f"Run config: sae_type={sae_type}, d_sae={d_sae}, k={top_k}, seed={seed}")

    # Find and download the SAE artifact
    artifact_name = f"{sae_type}-sae-d{d_sae}-k{top_k}-seed{seed}"
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
        sae_filename = f"{sae_type}_sae_d{d_sae}_k{top_k}_lr{lr}_seed{seed}_{model_name}.pt"
        sae_path = os.path.join('../sae_checkpoints/sweep_runs', sae_filename)
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE not found at: {sae_path}")
        
        print(f"Loading SAE from local: {sae_path}")
    
    result = load_sae_checkpoint(sae_path, d_model=None, device=device)
    checkpoint = result["checkpoint"]
    dict_size = result["config"].get("dict_size", result["config"].get("d_sae", d_sae))

    print(f"✓ Loaded {sae_type} SAE: d_sae={dict_size}")
    print(f"  Final loss: {checkpoint.get('final_loss', 'N/A')}")
    print(f"  Final L0: {checkpoint.get('final_l0', 'N/A')}")

    return {**result, "run_config": run_config}


def compare_sweep_runs(project=None,
                       sweep_id="wmhceuqf"):
    """
    Fetch summary statistics for all runs in a sweep.
    
    Args:
        project: W&B project path
        sweep_id: W&B sweep ID
    
    Returns:
        pandas DataFrame with run statistics
    """
    if project is None:
        raise ValueError("project must be provided as 'entity/project'")

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


def select_checkpoints(paths, use_best=False):
    """
    Filter checkpoint paths to avoid duplicating final/best pairs.
    
    When multiple checkpoints exist for the same configuration (e.g., one final
    checkpoint and one best-validation-loss checkpoint), this function selects
    which variant(s) to keep based on the use_best flag.
    
    Args:
        paths: List of checkpoint paths to filter
        use_best: If True, prefer best-val-loss checkpoints over final.
                 If False (default), keep only final checkpoints.
    
    Returns:
        tuple: (selected_paths, using_best_set)
            - selected_paths: Filtered list of checkpoint paths
            - using_best_set: Set of paths that are "best" variants
                             (empty if use_best=False)
    
    Examples:
        >>> paths = ['sae_d128_k3_lr0.0003_seed44_2layer_100dig_64d.pt',
        ...          'sae_d128_k3_lr0.0003_seed44_2layer_100dig_64d_best_0.9.pt']
        >>> selected, using_best = select_checkpoints(paths, use_best=False)
        >>> selected  # Only the final checkpoint
        ['sae_d128_k3_lr0.0003_seed44_2layer_100dig_64d.pt']
        >>> selected, using_best = select_checkpoints(paths, use_best=True)
        >>> selected  # Prefers the best variant
        ['sae_d128_k3_lr0.0003_seed44_2layer_100dig_64d_best_0.9.pt']
    """
    def _canonical(p):
        return re.sub(r'_best(?=_)', '', Path(p).stem)

    groups = defaultdict(dict)
    for p in paths:
        key = _canonical(p)
        tag = 'best' if '_best_' in Path(p).stem else 'final'
        groups[key][tag] = p

    selected, using_best_set = [], set()
    for key in sorted(groups):
        variants = groups[key]
        if use_best and 'best' in variants:
            selected.append(variants['best'])
            using_best_set.add(variants['best'])
        elif 'final' in variants:
            selected.append(variants['final'])
        elif 'best' in variants:  # only best variant exists
            selected.append(variants['best'])
            using_best_set.add(variants['best'])
    
    return selected, using_best_set
