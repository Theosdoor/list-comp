#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for grid search and hyperparameter sweep scripts
"""
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../..')
from transformer_lens import HookedTransformer
from model_scripts.model_utils import configure_runtime, make_model


def pick_device(explicit: str = "auto") -> torch.device:
    """Pick device for training"""
    if explicit != "auto":
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_batch(
    batch_size: int,
    n_digits: int,
    list_len: int,
    device: torch.device,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of list comparison examples.
    
    Returns:
      inputs  [B, T]
      targets [B, T]
    where T = 2L + 1 and evaluation uses targets[:, L+1:].
    """
    pad = n_digits
    sep = n_digits + 1
    T = 2 * list_len + 1

    digits = torch.randint(
        0, n_digits, (batch_size, list_len), generator=rng, device=device
    )

    inputs = torch.full((batch_size, T), pad, dtype=torch.long, device=device)
    targets = torch.full((batch_size, T), sep, dtype=torch.long, device=device)

    inputs[:, :list_len] = digits
    inputs[:, list_len] = sep

    targets[:, :list_len] = digits
    targets[:, list_len] = sep
    targets[:, list_len + 1:] = digits
    return inputs, targets


def make_validation_set(
    n_examples: int,
    n_digits: int,
    list_len: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a fixed validation set with a specific seed"""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    X, Y = make_batch(n_examples, n_digits, list_len, device, g)
    return X, Y


def eval_accuracy(
    model: HookedTransformer,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    list_len: int,
    batch_size: int = 1024,
) -> float:
    """
    Evaluate model accuracy on validation set.
    
    Args:
        model: The model to evaluate
        val_inputs: Validation inputs [N, T]
        val_targets: Validation targets [N, T]
        list_len: Length of the list
        batch_size: Batch size for evaluation
    
    Returns:
        Accuracy as a float in [0, 1]
    """
    model.eval()
    device = next(model.parameters()).device
    hits = 0
    tots = 0
    with torch.no_grad():
        for i in range(0, val_inputs.size(0), batch_size):
            xb = val_inputs[i: i + batch_size].to(device)
            yb = val_targets[i: i + batch_size].to(device)
            logits = model(xb)[:, list_len + 1:, :]  # [B, L, V]
            preds = logits.argmax(dim=-1)
            gold = yb[:, list_len + 1:]
            hits += (preds == gold).sum().item()
            tots += preds.numel()
    return hits / max(1, tots)


def build_model(
    list_len: int,
    n_digits: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    use_ln: bool = False,
    use_bias: bool = False,
    use_wv: bool = False,
    use_wo: bool = False,
    attn_only: bool = True,
    device: torch.device = None,
    seed: int = 0,
) -> HookedTransformer:
    """
    Build a model with the given architecture.
    
    Args:
        list_len: Length of the input list
        n_digits: Number of digits (vocabulary size - 2)
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        use_ln: Whether to use layer normalization
        use_bias: Whether to use bias terms
        use_wv: Whether to learn W_V (else freeze to identity)
        use_wo: Whether to learn W_O (else freeze to identity)
        attn_only: Whether to use attention-only (no MLP)
        device: Device to place model on
        seed: Random seed for initialization
    
    Returns:
        HookedTransformer model
    """
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    vocab = n_digits + 2
    seq_len = 2 * list_len + 1
    
    if device is None:
        device = pick_device()
    
    configure_runtime(list_len=list_len, seq_len=seq_len, vocab=vocab, device=device, seed=seed)
    model = make_model(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        ln=use_ln,
        use_bias=use_bias,
        use_wv=use_wv,
        use_wo=use_wo,
        attn_only=attn_only,
    )
    return model
