"""Model loading, saving, and evaluation utilities."""

import os
import torch

from ..utils.runtime import _RUNTIME
from .transformer import make_model

__all__ = [
    "accuracy",
    "save_model",
    "load_model",
    "infer_model_config",
]

# metrics
def accuracy(m, val_dl, list_len=None, device=None):
    if list_len is None:
        list_len = _RUNTIME.list_len
    if device is None:
        device = _RUNTIME.device
    assert list_len is not None, "list_len must be provided or configured via configure_runtime()"
    m.eval()
    hits = tots = 0
    with torch.no_grad():
        for inputs, targets in val_dl:
            logits = m(inputs.to(device))[:, list_len + 1 :]  # (batch, 2, vocab)
            preds = logits.argmax(-1)
            hits += (preds == targets[:, list_len + 1 :].to(device)).sum().item()
            tots += preds.numel()
    return hits / tots

# ----- Model saving / loading helpers ------
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, **model_kwargs):
    """Load weights into a freshly constructed model.

    Accepts any kwargs for make_model. If seq_len/list_len/vocab/device are omitted,
    the configured runtime values will be used.
    """
    device = model_kwargs.get("device", _RUNTIME.device)
    print("Loading model from", path)
    model = make_model(**model_kwargs)
    model.load_state_dict(
        torch.load(path, map_location=device)
    )  # map weights to target device
    model.eval()
    return model

def infer_model_config(path, device=None):
    """Infer model configuration from a checkpoint file.
    
    Args:
        path: Path to the checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        dict with keys: d_model, n_layers, n_heads, d_vocab, n_ctx, list_len,
                        attn_only, use_ln, use_bias, use_wv, use_wo
    """
    if device is None:
        device = _RUNTIME.device or "cpu"
    
    checkpoint = torch.load(path, map_location=device)
    
    # Infer d_model from W_Q shape: (n_heads, d_model, d_head)
    d_model = checkpoint['blocks.0.attn.W_Q'].shape[1]
    
    # Infer n_heads from W_Q shape: (n_heads, d_model, d_head)
    n_heads = checkpoint['blocks.0.attn.W_Q'].shape[0]
    
    # Infer n_layers by counting block keys
    n_layers = sum(1 for k in checkpoint.keys() if k.endswith('.attn.W_Q'))
    
    # Infer d_vocab from embed.W_E shape: (d_vocab, d_model)
    d_vocab = checkpoint['embed.W_E'].shape[0]
    
    # Infer n_ctx from pos_embed.W_pos shape: (n_ctx, d_model)
    n_ctx = checkpoint['pos_embed.W_pos'].shape[0]
    
    # Derive list_len from n_ctx: n_ctx = list_len * 2 + 1 => list_len = (n_ctx - 1) // 2
    # Validate that n_ctx is odd (required by the formula)
    if n_ctx % 2 == 0:
        raise ValueError(f"Invalid n_ctx={n_ctx}: expected odd value (n_ctx = list_len * 2 + 1)")
    list_len = (n_ctx - 1) // 2
    
    # Validate the relationship holds exactly
    expected_n_ctx = list_len * 2 + 1
    if n_ctx != expected_n_ctx:
        raise ValueError(f"n_ctx={n_ctx} does not match expected value {expected_n_ctx} for list_len={list_len}")
    
    # Infer attn_only: check if MLP weights exist
    attn_only = 'blocks.0.mlp.W_in' not in checkpoint
    
    # Infer use_ln from presence of non-zero layer norm weights
    use_ln = 'blocks.0.ln1.w' in checkpoint and (checkpoint['blocks.0.ln1.w'].abs().sum() > 0).item()
    
    # Infer use_bias from whether attention biases are non-zero
    use_bias = 'blocks.0.attn.b_Q' in checkpoint and (checkpoint['blocks.0.attn.b_Q'].abs().sum() > 0).item()
    
    # Infer use_wv: check if W_V differs from identity-like pattern
    W_V = checkpoint['blocks.0.attn.W_V']  # (n_heads, d_model, d_head)
    identity_slice = torch.eye(d_model, d_model // n_heads).unsqueeze(0).expand(n_heads, -1, -1)
    use_wv = not torch.allclose(W_V, identity_slice.to(W_V.device), atol=1e-5)
    
    # Infer use_wo: check if W_O differs from identity-like pattern
    W_O = checkpoint['blocks.0.attn.W_O']  # (n_heads, d_head, d_model)
    identity_slice_o = torch.eye(d_model // n_heads, d_model).unsqueeze(0).expand(n_heads, -1, -1)
    use_wo = not torch.allclose(W_O, identity_slice_o.to(W_O.device), atol=1e-5)
    
    config = {
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'd_vocab': d_vocab,
        'n_ctx': n_ctx,
        'list_len': list_len,
        'attn_only': attn_only,
        'use_ln': use_ln,
        'use_bias': use_bias,
        'use_wv': use_wv,
        'use_wo': use_wo,
    }
    print(f"Inferred config: {config}")
    return config
