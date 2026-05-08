"""
Notebook utilities for loading models and SAEs with consistent configuration.
"""

import torch
from pathlib import Path
from src.sae.loading import load_sae_checkpoint
from ..models.utils import load_model as _load_model
from ..models.transformer import parse_model_name_safe
from ..utils.runtime import configure_runtime


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_notebook(seed=42, disable_grad=True):
    """
    Common notebook setup: set device, seed, and gradient settings.
    Returns the device string ('cuda', 'mps', or 'cpu').
    """
    import numpy as np

    device = get_device()
    print(f"Using device: {device}")

    if disable_grad:
        torch.set_grad_enabled(False)

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
    use_wo=False,
):
    """
    Load a transformer model with standard configuration.

    Args:
        model_name: Name of the model file (e.g., '2layer_100dig_64d')
        device: Device to load model on (auto-detected if None)
        models_dir: Directory containing model files (defaults to project root/models)
        n_heads, ln, use_bias, use_wv, use_wo: Architecture flags

    Returns:
        (model, config) where config has: d_model, n_layers, n_digits,
        list_len, sep_token_index, n_heads
    """
    if device is None:
        device = get_device()

    if models_dir is None:
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models"
    else:
        models_dir = Path(models_dir)

    model_cfg = parse_model_name_safe(model_name)
    list_len = model_cfg.list_len
    n_digits = model_cfg.n_digits

    configure_runtime(
        list_len=list_len,
        seq_len=2 * list_len + 1,
        vocab=n_digits + 2,
        device=device,
    )

    model_path = models_dir / (model_name + ".pt")
    model = _load_model(
        str(model_path),
        n_layers=model_cfg.n_layers,
        n_heads=n_heads,
        d_model=model_cfg.d_model,
        ln=ln,
        use_bias=use_bias,
        use_wv=use_wv,
        use_wo=use_wo,
    )

    print(f"✓ Loaded model from {model_path}")

    config = {
        "d_model": model_cfg.d_model,
        "n_layers": model_cfg.n_layers,
        "n_digits": n_digits,
        "list_len": list_len,
        "sep_token_index": list_len,
        "n_heads": n_heads,
    }

    return model, config


def load_sae(sae_path, d_model, device=None):
    """
    Load a Sparse Autoencoder from a .pt checkpoint.

    Args:
        sae_path: Relative path from project root to a .pt file
        d_model: Dimension of model activations
        device: Device to load on (auto-detected if None)

    Returns:
        (sae, config) where config contains dict_size, d_sae, act_mean,
        and any other fields from the checkpoint config
    """
    if device is None:
        device = get_device()

    sae_path = Path(sae_path)

    if sae_path.suffix != ".pt":
        raise ValueError(f"Expected a .pt file, got: {sae_path}")
    if not sae_path.exists():
        raise FileNotFoundError(f"No SAE checkpoint found at {sae_path}")

    result = load_sae_checkpoint(sae_path, d_model=d_model, device=device)
    sae = result["sae"]
    config = result["config"]
    d_sae = config["d_sae"]
    sae_type = config.get("sae_type", "btk")

    print(f"✓ Loaded {sae_type} SAE from {sae_path} (dict_size={d_sae})")

    return sae, config
