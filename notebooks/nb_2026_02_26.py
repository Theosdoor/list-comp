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
# get pretty example for paper

# test_egs = [(28,90)] # cherry picked eg. of successful swap
test_egs = [(75,32), (32,75)]

results = feature_steering_experiment(
    model, sae, act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, 
    d2_all=d2_all, 
    sae_acts_all=sae_acts_all, 
    dataset=all_ds,
    test_pairs=test_egs,
    # sample_step_size=0.005,
    # transpose=True
)

# %%
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
# feat_zero reverse-pair analysis — stratified by alpha_diff sign
#
# Hypothesis: feature 30 tracks alpha_diff = alpha_d1 - alpha_d2 (SEP attention).
# When alpha_diff < 0 for (d1,d2), the feature doesn't fire (feat_zero).
# The reversed pair (d2,d1) then has alpha_diff > 0 and likely succeeds.

# Build a lookup: (d1, d2) -> (alpha_d1, alpha_d2, alpha_diff)
# alpha_d1_all / alpha_d2_all are tensors of shape (N,)
# d1_all / d2_all are tensors of shape (N,) with the actual digit values
d1_np = d1_all.cpu().numpy()
d2_np = d2_all.cpu().numpy()
alpha_d1_np = alpha_d1_all.cpu().numpy()
alpha_d2_np = alpha_d2_all.cpu().numpy()

attn_df = pd.DataFrame({
    'd1': d1_np.astype(int),
    'd2': d2_np.astype(int),
    'alpha_d1': alpha_d1_np,
    'alpha_d2': alpha_d2_np,
    'alpha_diff': alpha_d1_np - alpha_d2_np,
})

# Merge attention onto swap_bounds
bounds = swap_bounds_df.copy()
bounds['failure_reason'] = bounds['failure_reason'].fillna('success')
bounds = bounds.merge(attn_df, on=['d1', 'd2'], how='left')

fz = bounds[bounds['failure_reason'] == 'feat_zero'].copy()

# For each feat_zero row, look up the reverse pair's failure reason
rev_bounds = pd.DataFrame({'d1': fz['d2'].values, 'd2': fz['d1'].values})
rev_info = rev_bounds.merge(
    bounds[['d1', 'd2', 'failure_reason', 'alpha_diff']],
    on=['d1', 'd2'], how='left', suffixes=('', '_rev')
)
fz = fz.reset_index(drop=True)
fz['rev_reason'] = rev_info['failure_reason'].fillna('missing').values
fz['alpha_diff_rev'] = rev_info['alpha_diff'].values  # should be ≈ -alpha_diff

# Stratify by sign of alpha_diff for the feat_zero input
fz['attn_dir'] = fz['alpha_diff'].apply(
    lambda x: 'alpha_d1 > alpha_d2' if x > 0 else ('alpha_d1 < alpha_d2' if x < 0 else 'equal')
)

print("=== feat_zero: alpha_diff distribution ===")
print(f"mean alpha_diff: {fz['alpha_diff'].mean():.4f}")
print(f"fraction with alpha_diff < 0: {(fz['alpha_diff'] < 0).mean():.3f}")
print(f"fraction with alpha_diff > 0: {(fz['alpha_diff'] > 0).mean():.3f}")
print()

print("=== Reverse-pair failure reasons for ALL feat_zero ===")
print(fz['rev_reason'].value_counts())
print()

for direction in ['alpha_d1 < alpha_d2', 'alpha_d1 > alpha_d2']:
    subset = fz[fz['attn_dir'] == direction]
    print(f"=== feat_zero where {direction} (n={len(subset)}) ===")
    print(subset['rev_reason'].value_counts())
    print()

# Quick sanity check: (75,32) and (32,75)
for pair in [(75, 32), (32, 75)]:
    row = bounds[(bounds['d1'] == pair[0]) & (bounds['d2'] == pair[1])]
    if len(row):
        r = row.iloc[0]
        print(f"({pair[0]},{pair[1]}): failure={r['failure_reason']}, alpha_diff={r['alpha_diff']:.4f} (alpha_d1={r['alpha_d1']:.4f}, alpha_d2={r['alpha_d2']:.4f})")
# %%
# Mutual feat_zero analysis: correctness + |alpha_diff| magnitude
#
# Two questions:
# 1. Are mutual-feat_zero pairs ones the model got wrong originally?
# 2. Do the 139 (alpha_d1 > alpha_d2 fwd, but rev also feat_zero) have small |alpha_diff|?

# Build original model correctness for all inputs
model.eval()
all_dl_ordered = DataLoader(all_ds, batch_size=512, shuffle=False)
correctness_rows = []
with torch.no_grad():
    for inputs, targets in all_dl_ordered:
        inputs = inputs.to(DEVICE)
        logits = model(inputs)
        preds = logits[:, LIST_LEN + 1:, :N_DIGITS].argmax(-1)
        tgt = targets[:, LIST_LEN + 1:]
        d1s = targets[:, 0].tolist()
        d2s = targets[:, 1].tolist()
        o1c = (preds[:, 0] == tgt[:, 0].to(DEVICE)).tolist()
        o2c = (preds[:, 1] == tgt[:, 1].to(DEVICE)).tolist()
        for d1, d2, c1, c2 in zip(d1s, d2s, o1c, o2c):
            if c1 and c2:
                label = 'both_correct'
            elif c1 or c2:
                label = 'partial'
            else:
                label = 'both_wrong'
            correctness_rows.append({'d1': int(d1), 'd2': int(d2), 'correctness': label})
correctness_df = pd.DataFrame(correctness_rows)

# Merge correctness onto fz
fz_corr = fz.merge(correctness_df, on=['d1', 'd2'], how='left')
fz_corr['mutual_fz'] = fz_corr['rev_reason'] == 'feat_zero'

print("=== Correctness breakdown for ALL feat_zero ===")
print(fz_corr['correctness'].value_counts())
print()
print("=== Correctness: mutual feat_zero (both directions fail) ===")
print(fz_corr[fz_corr['mutual_fz']]['correctness'].value_counts())
print()
print("=== Correctness: feat_zero whose reverse succeeds ===")
print(fz_corr[fz_corr['rev_reason'] == 'success']['correctness'].value_counts())
print()

# Q2: are mutual-feat_zero (alpha_d1>alpha_d2) cases just near |alpha_diff|≈0?
mutual_hi = fz_corr[fz_corr['mutual_fz'] & (fz_corr['attn_dir'] == 'alpha_d1 > alpha_d2')]
mutual_lo = fz_corr[fz_corr['mutual_fz'] & (fz_corr['attn_dir'] == 'alpha_d1 < alpha_d2')]
rev_success = fz_corr[fz_corr['rev_reason'] == 'success']

print("=== |alpha_diff| stats by subgroup ===")
for label, subset in [
    ('mutual fz  (alpha_d1 > alpha_d2, n=%d)' % len(mutual_hi), mutual_hi),
    ('mutual fz  (alpha_d1 < alpha_d2, n=%d)' % len(mutual_lo), mutual_lo),
    ('rev=success               (n=%d)' % len(rev_success), rev_success),
]:
    ad = subset['alpha_diff'].abs()
    print(f"{label}  |alpha_diff|: mean={ad.mean():.4f}, median={ad.median():.4f}, min={ad.min():.4f}, max={ad.max():.4f}")
print()

# Distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (label, subset, color) in zip(axes, [
    ('mutual feat_zero', fz_corr[fz_corr['mutual_fz']], 'tomato'),
    ('rev=success', rev_success, 'steelblue'),
]):
    sns.histplot(subset['alpha_diff'].abs(), bins=40, ax=ax, color=color)
    ax.set_title(f'|alpha_diff| — {label} (n={len(subset)})')
    ax.set_xlabel('|alpha_diff|')
plt.suptitle('Do mutual-feat_zero inputs have smaller |alpha_diff|?', y=1.02)
plt.tight_layout()
plt.show()
# %%
