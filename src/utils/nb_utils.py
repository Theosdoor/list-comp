"""
Notebook utilities for loading models and SAEs with consistent configuration.
"""

import os
import sys
import torch
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

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
    models_dir="../models",
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
        models_dir: Directory containing model files
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
    model_path = os.path.join(models_dir, model_name + ".pt")
    model = _load_model(
        model_path,
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
    sae_name,
    d_model,
    device=None,
    sae_dir="../results/sae_models"
):
    """
    Load a Sparse Autoencoder (SAE) from checkpoint.
    
    Args:
        sae_name: Name of the SAE file (e.g., 'sae_d100_k4_50ksteps_2layer_100dig_64d.pt')
        d_model: Dimension of model activations
        device: Device to load SAE on (auto-detected if None)
        sae_dir: Directory containing SAE checkpoints
    
    Returns:
        tuple: (sae, sae_config) where sae_config contains:
            - dict_size (d_sae): SAE dictionary size
            - k (top_k): Number of active features
            - Additional fields from checkpoint config
    """
    if device is None:
        device = get_device()
    
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
    
    print(f"✓ Loaded SAE from {sae_path}")
    print(f"  - Dictionary size: {d_sae}")
    print(f"  - Top-K: {top_k}")
    
    # Return SAE and config
    config = {
        'dict_size': d_sae,
        'd_sae': d_sae,
        'k': top_k,
        'top_k': top_k,
        **sae_cfg  # Include any additional config from checkpoint
    }
    
    return sae, config
