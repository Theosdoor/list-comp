"""
Notebook utilities for loading models and SAEs with consistent configuration.
"""

import os
import sys
import torch
from pathlib import Path
from src.sae.loading import instantiate_sae_from_cfg

from ..models.utils import load_model as _load_model
from ..models.transformer import parse_model_name_safe
from ..utils.runtime import configure_runtime


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_notebook(seed=42, disable_grad=True):
    """
    Common notebook setup: set device, seed, and gradient settings.
    
    Args:
        seed: Random seed for reproducibility
        disable_grad: Whether to disable gradients (default True for analysis)
    
    Returns:
        device: The device string ('cuda', 'mps', or 'cpu')
    """
    import numpy as np
    
    device = get_device()
    print(f"Using device: {device}")
    
    if disable_grad:
        torch.set_grad_enabled(False)
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    return device


def load_transformer_model(
    model_name,
    device=None,
    models_dir=None,
    n_heads=1,
    ln=False,
    use_bias=False,
    use_wv=False,
    use_wo=False
):
    """
    Load a transformer model with standard configuration.
    
    Args:
        model_name: Name of the model file (e.g., '2layer_100dig_64d')
        device: Device to load model on (auto-detected if None)
        models_dir: Directory containing model files (defaults to project root/models)
        n_heads: Number of attention heads
        ln: Whether to use layer normalization
        use_bias: Whether to use bias terms
        use_wv: Whether to use separate value weights
        use_wo: Whether to use output weights
    
    Returns:
        tuple: (model, model_config) where model_config contains:
            - d_model: Model dimension
            - n_layers: Number of layers
            - n_digits: Vocabulary size (digits)
            - list_len: List length for the task
            - sep_token_index: Position of separator token
    """
    if device is None:
        device = get_device()
    
    # Determine models directory
    if models_dir is None:
        project_root = Path(__file__).parent.parent.parent  # src/utils/ -> project root
        models_dir = project_root / "models"
    else:
        models_dir = Path(models_dir)
    
    # Parse model configuration
    model_cfg = parse_model_name_safe(model_name)
    
    # Derive task configuration
    list_len = 2  # Default for current tasks
    n_digits = model_cfg.n_digits
    sep_token_index = list_len  # SEP at position list_len
    
    # Setup runtime
    configure_runtime(
        list_len=list_len,
        seq_len=2 * list_len + 1,  # [d1, d2, SEP, o1, o2]
        vocab=n_digits + 2,  # digits + MASK + SEP
        device=device
    )
    
    # Load model
    model_path = models_dir / (model_name + ".pt")
    model = _load_model(
        str(model_path),
        n_layers=model_cfg.n_layers,
        n_heads=n_heads,
        d_model=model_cfg.d_model,
        ln=ln,
        use_bias=use_bias,
        use_wv=use_wv,
        use_wo=use_wo
    )
    
    print(f"✓ Loaded model from {model_path}")
    
    # Return model and config dict
    config = {
        'd_model': model_cfg.d_model,
        'n_layers': model_cfg.n_layers,
        'n_digits': n_digits,
        'list_len': list_len,
        'sep_token_index': sep_token_index,
        'n_heads': n_heads,
    }
    
    return model, config


def load_sae(
    sae_path,
    d_model,
    device=None,
    sae_dir=None
):
    """
    Load a Sparse Autoencoder (SAE) from checkpoint.
    
    Args:
        sae_path: Path to SAE checkpoint. Can be:
            - A folder containing a .pt file (recommended)
            - A .pt file directly (legacy, will warn)
            - A relative path from sae_dir (if sae_dir is provided)
            - A relative path from sae_checkpoints/ (if sae_dir is None)
        d_model: Dimension of model activations
        device: Device to load SAE on (auto-detected if None)
        sae_dir: Directory containing SAE checkpoints (defaults to project root/sae_checkpoints).
                 Ignored if sae_path is absolute or if sae_path already exists.
    
    Returns:
        tuple: (sae, sae_config) where sae_config contains:
            - dict_size (d_sae): SAE dictionary size
            - k (top_k): Number of active features
            - Additional fields from checkpoint config
    """
    if device is None:
        device = get_device()
    
    # Convert to Path object
    sae_path = Path(sae_path)
    
    # Resolve project root
    if sae_dir is None:
        project_root = Path(__file__).parent.parent.parent  # src/utils/ -> project root
        sae_dir = project_root / "sae_checkpoints"
    else:
        project_root = sae_dir.parent if isinstance(sae_dir, Path) else Path(sae_dir).parent
        sae_dir = Path(sae_dir)
    
    # If not absolute, resolve relative to sae_dir or project_root
    if not sae_path.is_absolute():
        # Check if path already starts with sae_checkpoints (avoid double-prefixing)
        sae_dir_name = sae_dir.name
        if sae_path.parts and sae_path.parts[0] == sae_dir_name:
            # Path already starts with 'sae_checkpoints', use it relative to project root
            sae_path = project_root / sae_path
        elif not sae_path.exists():
            # Path doesn't exist and doesn't start with sae_checkpoints, so prepend sae_dir
            sae_path = sae_dir / sae_path
        # else: Path exists as-is (relative to cwd), keep it as-is
    
    # If it's a directory, find the .pt file inside
    if sae_path.is_dir():
        pt_files = list(sae_path.glob("*.pt"))
        if len(pt_files) == 0:
            raise ValueError(f"No .pt files found in SAE directory: {sae_path}")
        if len(pt_files) > 1:
            raise ValueError(f"Multiple .pt files found in SAE directory {sae_path}: {[f.name for f in pt_files]}. Please specify the exact file.")
        sae_path = pt_files[0]
    
    # If it ends with .pt, thats fine!
    if sae_path.suffix == ".pt":
        # Check if there's a parent folder that might be the intended SAE folder
        parent_dir = sae_path.parent
        other_pt_files = [f for f in parent_dir.glob("*.pt") if f != sae_path]
        # if len(other_pt_files) == 0:
        #     print(f"⚠ Note: You passed a .pt file directly. In the future, pass the folder instead: {parent_dir}")
    
    # Load checkpoint
    checkpoint = torch.load(str(sae_path), map_location=device, weights_only=False)
    
    # Extract config
    sae_cfg = checkpoint.get("cfg", {})
    d_sae = sae_cfg.get("dict_size", sae_cfg.get("d_sae", 256))
    sae_type = sae_cfg.get("sae_type", "btk")

    # Create SAE instance using shared dispatch
    sae = instantiate_sae_from_cfg(sae_cfg, d_model, device)

    # Load state dict (handle legacy BTK format)
    state_dict = checkpoint["state_dict"]
    if sae_type == "btk" and "W_enc" in state_dict:
        state_dict = {
            "encoder.weight": state_dict["W_enc"].T,
            "encoder.bias": state_dict["b_enc"],
            "decoder.weight": state_dict["W_dec"],
            "decoder.bias": state_dict["b_dec"],
        }
    sae.load_state_dict(state_dict)

    act_mean = checkpoint.get("act_mean", torch.zeros(d_model))

    print(f"✓ Loaded {sae_type} SAE from {sae_path}")
    print(f"  - Dictionary size: {d_sae}")

    return sae, {'dict_size': d_sae, 'd_sae': d_sae, 'act_mean': act_mean, **sae_cfg}
