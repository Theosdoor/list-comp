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
LIST_LEN = 2 # [d1, d2]
SEQ_LEN = LIST_LEN * 2 + 1 # [d1, d2, SEP, o1, o2]

N_DIGITS = 100
DIGITS = list(range(N_DIGITS)) # 100 digits from 0 to 99
MASK = N_DIGITS # special masking token for o1 and o2
SEP = N_DIGITS+1 # special seperator token for the model to think about the input (+1 to avoid confusion with the last digit)
VOCAB = len(DIGITS) + 2  # + the special tokens

D_MODEL = 64
N_HEAD = 1
N_LAYER = 2
USE_LN = True # use layer norm in model
USE_BIAS = True # use bias in model
USE_WV = True # use value matrix in attn (False = freeze to identity)
USE_WO = True # use output matrix in attn (False = freeze to identity)

LEARNING_RATE = 1e-3 # default 1e-3
WEIGHT_DECAY = 0.01 # default 0.01
MAX_TRAIN_STEPS = 50_000 # max training steps
USE_CHECKPOINTING = False # whether to use checkpointing for training

RUN_TS = datetime.now().strftime("%Y%m%d-%H%M%S")
tf_string = ''.join('T' if b else 'F' for b in [USE_LN, USE_BIAS, USE_WV, USE_WO])
MODEL_NAME = f'{N_LAYER}layer_{N_DIGITS}dig_{D_MODEL}d_LBVO-{tf_string}_{RUN_TS}'
# Construct path relative to project root (parent of model_scripts/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", MODEL_NAME + ".pt")

# --- dataset --- (not necessary as we fix seed?)
# DATASET_NAME = None # None ==> generate new one
# listlen2_digits10_dupes
# listlen2_digits10_nodupes
# listlen2_digits100_dupes_traindupesonly
# listlen2_digits100_dupes
# listlen2_digits100_nodupes

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
torch.manual_seed(0)

# Provide runtime config so we don't need to thread constants everywhere
configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV)

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
    model = make_model(
        n_layers=N_LAYER,
        n_heads=N_HEAD,
        d_model=D_MODEL,
        ln=USE_LN,
        use_bias=USE_BIAS,
        use_wv=USE_WV,
        use_wo=USE_WO,
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
