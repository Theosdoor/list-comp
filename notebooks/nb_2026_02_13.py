# %%
# SETUP

from torch._numpy import False_
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy import stats
import wandb

# Import notebook utilities
from src.utils.nb_utils import setup_notebook, load_transformer_model, load_sae

# Import project utilities
from src.utils.runtime import configure_runtime
from src.models.transformer import parse_model_name_safe, build_attention_mask
from src.models.utils import load_model
from src.data.datasets import get_dataset
from src.sae.sae_analysis import *

# Setup device and seeds
DEVICE = setup_notebook(seed=42)

# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
SAE_NAME = "sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt" # MSE: 0.0042, Recon Acc: 0.8688 (old - not as high accuracy as below)

# Output Config
SAVE_RESULTS = False
SAVE_DIR = "../results/sae_results/" if SAVE_RESULTS else None

# --- Load Models ---
model, model_cfg = load_transformer_model(MODEL_NAME, device=DEVICE)

# Extract config for convenience
D_MODEL = model_cfg['d_model']
N_LAYERS = model_cfg['n_layers']
N_HEADS = model_cfg['n_heads']
LIST_LEN = model_cfg['list_len']
N_DIGITS = model_cfg['n_digits']
SEP_TOKEN_INDEX = model_cfg['sep_token_index']

# Load SAE
sae, sae_cfg = load_sae(SAE_NAME, D_MODEL, device=DEVICE)
D_SAE = sae_cfg['dict_size']
TOP_K = sae_cfg['k']

# Load activation mean from checkpoint (for centering)
SAE_PATH = os.path.join('../results/sae_models', SAE_NAME)
sae_checkpoint = torch.load(SAE_PATH, map_location=DEVICE, weights_only=False)
act_mean = sae_checkpoint["act_mean"].to(DEVICE)

# Load validation dataset and collect SAE activations
train_ds, val_ds = get_dataset(
    n_digits=N_DIGITS,
    list_len=LIST_LEN,
    no_dupes=False,
    train_dupes_only=False
)
# concat both
all_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)
all_dl = DataLoader(all_ds, batch_size=128, shuffle=False)

# Collect SAE activations for ALL data (not getting acccuracy so it's fine to incl train)
d1_all, d2_all, sae_acts_all = collect_sae_activations(
    model=model,
    sae=sae,
    val_dl=all_dl, 
    act_mean=act_mean,
    layer_idx=0,
    sep_idx=SEP_TOKEN_INDEX,
    device=DEVICE
)

# get data for analysis

# Extract attention weights from SEP token to input positions
alpha_d1_all, alpha_d2_all = collect_attention_weights(
    model=model,
    dataloader=all_dl,
    sep_idx=SEP_TOKEN_INDEX,
    device=DEVICE
)

# Identify special features
special_features_info = identify_special_features(
    sae_acts_all=sae_acts_all,
    alpha_d1_all=alpha_d1_all,
    alpha_d2_all=alpha_d2_all,
    threshold=0.5
)

feature_firing_freq = (sae_acts_all > 0).float().mean(dim=0).numpy()
active_features = np.where(feature_firing_freq > 0)[0]
n_active = len(active_features)

# Sort by firing frequency
sorted_indices = np.argsort(feature_firing_freq)[::-1]
top_n = min(30, n_active)


# %%
# see rough results
SPECIAL_FEAT_IDX = 30

results=feature_steering_experiment(
    model=model, sae=sae, act_mean=act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
    dataset=all_ds,
    sep_idx=SEP_TOKEN_INDEX,
    n_digits=N_DIGITS,
    save_dir=None
)

# %%
# bisection for exact crossover points (to 3dp)

coarse_results = feature_steering_experiment(
    model=model, sae=sae, act_mean=act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
    dataset=all_ds,
    sep_idx=SEP_TOKEN_INDEX,
    n_digits=N_DIGITS,
    plot=False,  # Don't plot since we're finding exact values
    scale_factors=np.linspace(0, 10, 20),  # Coarse sampling just to find regions
    save_dir=None
)

# Perform crossover analysis with bisection for exact values
print("\n" + "="*60)
print("CROSSOVER ANALYSIS (exact to 3dp)")
print("="*60)

for i, result in enumerate(coarse_results):
    d1_val = result['d1']
    d2_val = result['d2']
    scale_factors = result['scales']
    all_logits_o1 = result['all_logits_o1']
    all_logits_o2 = result['all_logits_o2']
    
    # Get data needed for bisection
    mask = (d1_all == d1_val) & (d2_all == d2_val)
    idx = torch.where(mask)[0][0].item()
    inputs_i = all_ds[idx][0].unsqueeze(0).to(DEVICE)
    z_orig = sae_acts_all[idx].clone().to(DEVICE)
    feat_orig = z_orig[SPECIAL_FEAT_IDX].item()
    
    # Find original output (at scale = 1.0)
    original_scale_idx = np.argmin(np.abs(scale_factors - 1.0))
    original_output_o1 = result['output_o1'][original_scale_idx]
    original_output_o2 = result['output_o2'][original_scale_idx]
    
    print(f"\n{'='*60}")
    print(f"Test Case {i+1}: Input ({d1_val}, {d2_val})")
    print(f"Original feature {SPECIAL_FEAT_IDX} activation: {feat_orig:.4f}")
    print(f"Original model output: ({original_output_o1}, {original_output_o2})")
    print(f"{'='*60}")
    
    # Extract d1 and d2 logits
    d1_logits_o1 = all_logits_o1[:, d1_val]
    d2_logits_o1 = all_logits_o1[:, d2_val]
    diff_o1 = d1_logits_o1 - d2_logits_o1
    
    d1_logits_o2 = all_logits_o2[:, d1_val]
    d2_logits_o2 = all_logits_o2[:, d2_val]
    diff_o2 = d1_logits_o2 - d2_logits_o2
    
    # Find where sign changes (crossover) for o1
    sign_changes_o1 = np.where(np.diff(np.sign(diff_o1)))[0]
    if len(sign_changes_o1) > 0:
        print(f"\n📍 O1: Found {len(sign_changes_o1)} crossover point(s)")
        for j, crossover_idx in enumerate(sign_changes_o1, 1):
            # Use bisection to find exact crossover (to 3dp)
            exact_scale = find_exact_crossover_bisection(
                model=model, sae=sae, act_mean=act_mean,
                feature_idx=SPECIAL_FEAT_IDX,
                inputs_i=inputs_i, z_orig=z_orig, feat_orig=feat_orig,
                d1_val=d1_val, d2_val=d2_val,
                scale_low=scale_factors[crossover_idx],
                scale_high=scale_factors[crossover_idx + 1],
                output_pos=-2,  # o1 position
                layer_idx=0, sep_idx=SEP_TOKEN_INDEX, n_digits=N_DIGITS,
                device=DEVICE
            )
            pred_o1 = result['output_o1'][crossover_idx]
            pred_o2 = result['output_o2'][crossover_idx]
            is_swapped = (pred_o1 == d2_val and pred_o2 == d1_val)
            swap_indicator = " SWAPPED!" if is_swapped else ""
            print(f"   Crossover #{j} at scale = {exact_scale:.3f} (3dp)")
            print(f"      d1 logit = {d1_logits_o1[crossover_idx]:.3f}, d2 logit = {d2_logits_o1[crossover_idx]:.3f}")
            print(f"      → Model output: ({pred_o1}, {pred_o2}){swap_indicator}")
    else:
        print(f"\n❌ O1: No crossover detected in range [{scale_factors[0]:.1f}, {scale_factors[-1]:.1f}]")
        if d1_logits_o1[0] > d2_logits_o1[0]:
            print("   d1 logit remains higher throughout")
        else:
            print("   d2 logit remains higher throughout")
    
    # Find where sign changes (crossover) for o2
    sign_changes_o2 = np.where(np.diff(np.sign(diff_o2)))[0]
    if len(sign_changes_o2) > 0:
        print(f"\n📍 O2: Found {len(sign_changes_o2)} crossover point(s)")
        for j, crossover_idx in enumerate(sign_changes_o2, 1):
            # Use bisection to find exact crossover
            exact_scale = find_exact_crossover_bisection(
                model=model, sae=sae, act_mean=act_mean,
                feature_idx=SPECIAL_FEAT_IDX,
                inputs_i=inputs_i, z_orig=z_orig, feat_orig=feat_orig,
                d1_val=d1_val, d2_val=d2_val,
                scale_low=scale_factors[crossover_idx],
                scale_high=scale_factors[crossover_idx + 1],
                output_pos=-1,  # o2 position
                layer_idx=0, sep_idx=SEP_TOKEN_INDEX, n_digits=N_DIGITS,
                device=DEVICE
            )
            pred_o1 = result['output_o1'][crossover_idx]
            pred_o2 = result['output_o2'][crossover_idx]
            is_swapped = (pred_o1 == d2_val and pred_o2 == d1_val)
            swap_indicator = " SWAPPED!" if is_swapped else ""
            print(f"   Crossover #{j} at scale = {exact_scale:.3f} (3dp)")
            print(f"      d1 logit = {d1_logits_o2[crossover_idx]:.3f}, d2 logit = {d2_logits_o2[crossover_idx]:.3f}")
            print(f"      → Model output: ({pred_o1}, {pred_o2}){swap_indicator}")
    else:
        print(f"\n❌ O2: No crossover detected in range [{scale_factors[0]:.1f}, {scale_factors[-1]:.1f}]")
        if d1_logits_o2[0] > d2_logits_o2[0]:
            print("   d1 logit remains higher throughout")
        else:
            print("   d2 logit remains higher throughout")
# %%
