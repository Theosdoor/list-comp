# %%
# SETUP
import os
import sys
import ast

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
from src.utils.nb_utils import *
from src.data.datasets import get_dataset
from src.sae import *  # Import all SAE analysis utilities

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

SPECIAL_FEAT_IDX = 30

# load crossover stuff
xovers_df = pd.read_csv(f'../results/xover/xovers_feat{SPECIAL_FEAT_IDX}.csv')
swap_bounds_df = pd.read_csv(f'../results/xover/swap_bounds_feat{SPECIAL_FEAT_IDX}.csv')
swap_results_df = pd.read_csv(f'../results/xover/swap_results_feat{SPECIAL_FEAT_IDX}.csv')


# %%
# CROSSOVER ANALYSIS: Load pre-computed results from GPU job

# Helper to parse list columns from CSV
def parse_list_column(df, col_name):
    """Parse string representation of lists back to actual lists."""
    df[col_name] = df[col_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

# Load crossovers
parse_list_column(xovers_df, 'o1_crossovers')
parse_list_column(xovers_df, 'o2_crossovers')

# Analyze crossover statistics
print(f"\nTotal inputs: {len(xovers_df)}")
print(f"Inputs with feature firing: {(xovers_df['feat_orig'] > 0).sum()}")
print(f"Inputs with no crossovers: {((xovers_df['n_o1_xover'] == 0) & (xovers_df['n_o2_xover'] == 0)).sum()}")
print(f"\nCrossover pattern distribution:")
print(xovers_df.groupby(['n_o1_xover', 'n_o2_xover']).size().to_frame('count'))
fires_no_xover = xovers_df[
    (xovers_df['feat_orig'] > 0) &
    (xovers_df['n_o1_xover'] == 0) &
    (xovers_df['n_o2_xover'] == 0) 
]
print(f"\nInputs where feature fires but no crossovers: {len(fires_no_xover)}")
# display(fires_no_xover.head(20))

# %%
# Display inputs where feature fires but no crossovers found
fires_no_xover = xovers_df[
    (xovers_df['feat_orig'] > 0) &
    (xovers_df['n_o1_xover'] == 0) &
    (xovers_df['n_o2_xover'] == 0) &
    (xovers_df['o1_failure_reason'] != 'd1_eq_d2')
]
print(f"\nFilterng out d1=d2 too: {len(fires_no_xover)}")
display(fires_no_xover.head(20))
# %%
# lets look at their graphs
# test_egs = [(55,76), (93,16), (61,26)]
test_egs = [(60,44)]

results = feature_steering_experiment(
    model, sae, act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, 
    d2_all=d2_all, 
    sae_acts_all=sae_acts_all, 
    dataset=all_ds,
    scale_range=[-1.0, 1.0],
    test_pairs=test_egs,
    sample_step_size=0.001
)

crossover_df = analyze_feature_crossovers(
    results=results,
    model=model, sae=sae, act_mean=act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
    dataset=all_ds,
    layer_idx=0,
    sep_idx=SEP_TOKEN_INDEX,
    n_digits=N_DIGITS,
    device=DEVICE,
    verbose=True
)
# %%
# Test the egs at different scales
for d1_val, d2_val in test_egs:
    mask = (d1_all == d1_val) & (d2_all == d2_val)
    idx = torch.where(mask)[0][0].item()
    inputs_i = all_ds[idx][0].unsqueeze(0).to(DEVICE)
    z_orig = sae_acts_all[idx].clone().to(DEVICE)
    feat_orig = z_orig[SPECIAL_FEAT_IDX].item()
    
    print(f"\n{'='*60}")
    print(f"Input: ({d1_val}, {d2_val}), Original feature {SPECIAL_FEAT_IDX}: {feat_orig:.4f}")
    print(f"{'='*60}")
    
    # Inspect at multiple scales
    custom_scales = [-0.7, -0.6, -0.5, -0.3, -0.01, 0.0]
    # custom_scales = [2.2,2.3]
    result_list, df = inspect_steered_outputs_batch(
        model=model, sae=sae, act_mean=act_mean,
        feature_idx=SPECIAL_FEAT_IDX,
        scales=custom_scales,
        inputs_i=inputs_i, z_orig=z_orig, feat_orig=feat_orig,
        d1_val=d1_val, d2_val=d2_val,
        layer_idx=0, sep_idx=SEP_TOKEN_INDEX, n_digits=N_DIGITS,
        device=DEVICE
    )
    print(df.to_string(index=False))

# %% [markdown]
# Let's focus on inputs that don't F30 can't steer (no swap bound) & see if another of the topk features activate it

# %%
test_egs = [(74,32)] # (75,32) is an input that doesnt activate f30 at all

