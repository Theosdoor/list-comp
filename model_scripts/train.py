# %% [markdown]
# ## Setup

# %%
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import copy
from datetime import datetime # for unique model naming

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import einops
import pandas as pd, itertools
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from model_utils import (
    configure_runtime,
    build_attention_mask,
    save_model,
    make_model,
    accuracy
)
from data import get_dataset

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# %% [markdown]
# ## Model

# %%
# ---------- parameters ----------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a list comparison transformer model")
    
    # Model architecture
    parser.add_argument("--n-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-digits", type=int, default=100, help="Number of digits (vocabulary size - 2)")
    parser.add_argument("--list-len", type=int, default=2, help="Length of input list (e.g., 2 for [d1,d2], 3 for [d1,d2,d3])")
    
    # Model features (flags)
    parser.add_argument("--ln", action="store_true", default=False, help="Use layer normalization")
    parser.add_argument("--bias", action="store_true", default=False, help="Use bias terms")
    parser.add_argument("--wv", action="store_true", default=False, help="Learn W_V (else freeze to identity)")
    parser.add_argument("--wo", action="store_true", default=False, help="Learn W_O (else freeze to identity)")
    parser.add_argument("--mlp", action="store_true", default=False, help="Include MLP layers")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-steps", type=int, default=50_000, help="Max training steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint", action="store_true", help="Save checkpoints during training")
    
    # Output
    parser.add_argument("--name", type=str, default=None, help="Custom model name (overrides auto-generated)")
    
    return parser.parse_args()

# Parse arguments (use defaults if running as notebook)
try:
    args = parse_args()
except SystemExit:
    # Running in notebook/interactive mode, use defaults
    args = argparse.Namespace(
        n_layers=2, n_heads=1, d_model=64, n_digits=100, list_len=2,
        ln=False, bias=False, wv=False, wo=False, mlp=False,
        lr=1e-3, weight_decay=0.01, max_steps=50_000, seed=0,
        checkpoint=False, name=None
    )

# Extract to module-level variables for compatibility
LIST_LEN = args.list_len
SEQ_LEN = LIST_LEN * 2 + 1  # [d1, ..., dn, SEP, o1, ..., on]

N_DIGITS = args.n_digits
DIGITS = list(range(N_DIGITS))
MASK = N_DIGITS
SEP = N_DIGITS + 1
VOCAB = len(DIGITS) + 2

D_MODEL = args.d_model
N_HEAD = args.n_heads
N_LAYER = args.n_layers
USE_LN = args.ln
USE_BIAS = args.bias
USE_WV = args.wv
USE_WO = args.wo
ATTN_ONLY = not args.mlp

LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
MAX_TRAIN_STEPS = args.max_steps
USE_CHECKPOINTING = args.checkpoint
SEED = args.seed

# Generate model name: L{layers}_H{heads}_D{d_model}_V{digits}[_flags]_timestamp
# Flags are only included if non-default (ln, bias, wv, wo, mlp)
RUN_TS = datetime.now().strftime("%y%m%d-%H%M%S")
base_name = f"L{N_LAYER}_H{N_HEAD}_D{D_MODEL}_V{N_DIGITS}"

# Add flags suffix only for non-default settings
flags = []
if LIST_LEN != 2: flags.append(f"len{LIST_LEN}")
if USE_LN: flags.append("ln")
if USE_BIAS: flags.append("bias")
if USE_WV: flags.append("wv")
if USE_WO: flags.append("wo")
if not ATTN_ONLY: flags.append("mlp")
flags_suffix = "_" + "-".join(flags) if flags else ""

MODEL_NAME = args.name if args.name else f"{base_name}{flags_suffix}_{RUN_TS}"

# Construct path relative to project root (parent of model_scripts/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", MODEL_NAME + ".pt")

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Provide runtime config so we don't need to thread constants everywhere
configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV, seed=SEED)

# %%
# ---------- mask ----------
# attention mask for [d1, d2, SEP, o1, o2] looks like this:
# -    d1    d2    SEP    o1    o2   (keys)
# d1   0    -inf   -inf  -inf  -inf
# d2   0    -inf   -inf  -inf  -inf
# SEP  0      0    -inf  -inf  -inf
# o1  -inf  -inf    0    -inf   -inf
# o2  -inf  -inf    0      0    -inf
# (queries)

# view mask
mask_bias, _ = build_attention_mask()
print(mask_bias.cpu()[0][0])

# %%
# ---------- dataset ----------
train_ds, val_ds = get_dataset(
    list_len=LIST_LEN, 
    n_digits=N_DIGITS, 
    train_split=0.8,
    mask_tok=MASK, # use MASK as mask token
    sep_tok=SEP, # use SEP as separator token
    )

train_batch_size = min(128, len(train_ds))
val_batch_size = min(256, len(val_ds))
train_dl = DataLoader(train_ds, train_batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, val_batch_size, drop_last=False)

print("Input:", train_ds[0][0])
print("Target:", train_ds[0][1])
print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")


# %%
def train(m, max_steps=10_000, early_stop_acc=0.999, checkpoints=False, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    dl = itertools.cycle(train_dl)  # infinite iterator
    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        inputs, targets = next(dl)
        # get logits/loss for output tokens only
        logits = m(inputs.to(DEV))[:, LIST_LEN+1:].reshape(-1, VOCAB) 
        loss = ce(logits, targets[:, LIST_LEN+1:].reshape(-1).to(DEV))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (step + 1) % 100 == 0:
            acc = accuracy(m, val_dl)
            if acc > early_stop_acc:
                print(f"Early stopping at step {step + 1} with accuracy {acc:.2%} >= {early_stop_acc:.2%}")
                break
            # Update tqdm bar w/ metrics
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2%}",
            })
            if checkpoints and (step+1) % 50_000 == 0:
                save_model(m, MODEL_PATH)
            
    print(f"Final accuracy: {accuracy(m, val_dl):.2%}")


# %%
# train and SAVE new model
acc = 0
while acc < 0.9:
    print(f"Training {MODEL_NAME}")
    print(f"  Config: {N_LAYER} layers, {N_HEAD} heads, d_model={D_MODEL}, LN={USE_LN}, bias={USE_BIAS}, WV={USE_WV}, WO={USE_WO}, attn_only={ATTN_ONLY}")
    model = make_model(
        n_layers=N_LAYER,
        n_heads=N_HEAD,
        d_model=D_MODEL,
        ln=USE_LN,
        use_bias=USE_BIAS,
        use_wv=USE_WV,
        use_wo=USE_WO,
        attn_only=ATTN_ONLY,
    )
    train(model, max_steps=MAX_TRAIN_STEPS, checkpoints=USE_CHECKPOINTING)
    acc = accuracy(model, val_dl)
    if acc > 0.8:
        MODEL_NAME_WITH_ACC = f'{MODEL_NAME}_acc{acc:.4f}'
        MODEL_PATH_WITH_ACC = os.path.join(_PROJECT_ROOT, "models", f"{MODEL_NAME_WITH_ACC}.pt")
        save_model(model, MODEL_PATH_WITH_ACC)

# %%
# --- Model Parameters Overview ---
m_for_overview = globals().get('model', None)
if m_for_overview is not None:
    print("--- Overview of Model Parameters ---")   
    total_params = 0
    trainable_params = 0

    # Use a formatted string for better alignment
    print(f"{'Parameter Name':<40} | {'Shape':<20} | {'Trainable':<10}")
    print("-" * 80)

    for name, param in m_for_overview.named_parameters():
        shape_str = str(tuple(param.shape))
        is_trainable = "Yes" if param.requires_grad else "No"
        total_params += param.numel()

        if not param.requires_grad:
            continue
        # Print only trainable parameters
        print(f"{name:<40} | {shape_str:<20} | {is_trainable:<10}")
        trainable_params += param.numel()

    print("-" * 80)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print("-" * 80)

# %%
