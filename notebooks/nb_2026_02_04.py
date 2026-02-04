#%% [markdown]
# interp saes

# %%
# SETUP

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
from src.sae.sae_analysis import (
    collect_sae_activations,
    create_feature_heatmaps,
    compute_reconstruction_metrics,
    identify_special_features,
    load_sae_from_wandb_run,
    compare_sweep_runs,
)

# Setup device and seeds
DEVICE = setup_notebook(seed=42)

# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
SAE_NAME = "sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d.pt"

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

# %%
# Load validation dataset and collect SAE activations
train_dataset, val_dataset = get_dataset(
    n_digits=N_DIGITS,
    list_len=LIST_LEN,
    no_dupes=False,
    train_dupes_only=False
)
val_dl = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Collect SAE activations for validation set
d1_all, d2_all, sae_acts_all = collect_sae_activations(
    model=model,
    sae=sae,
    val_dl=val_dl,
    act_mean=act_mean,
    layer_idx=0,
    sep_idx=SEP_TOKEN_INDEX,
    device=DEVICE
)

# %%
# basic metrics

# L0 Sparsity
l0 = (sae_acts_all > 0).float().sum(dim=1).mean()
print(f"Average L0 (active features per sample): {l0:.2f}")

# Dead features
dead_features = (sae_acts_all.sum(dim=0) == 0).sum().item()
print(f"Dead features: {dead_features} / {D_SAE} ({100*dead_features/D_SAE:.1f}%)")

# Feature firing rates
firing_rate = (sae_acts_all > 0).float().mean(dim=0)
print(f"Firing rate range: [{firing_rate[firing_rate > 0].min():.4f}, {firing_rate.max():.4f}]")

# %%
# Feature heatmaps
fig = create_feature_heatmaps(
    d1_all=d1_all,
    d2_all=d2_all,
    sae_acts_all=sae_acts_all,
    n_digits=N_DIGITS,
    figsize=(20, 20)
)
plt.tight_layout()
if SAVE_RESULTS:
    plt.savefig(f"{SAVE_DIR}sae_feature_heatmaps.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# Reconstruction metrics
recon_metrics = compute_reconstruction_metrics(
    model=model,
    sae=sae,
    val_dl=val_dl,
    act_mean=act_mean,
    layer_idx=0,
    sep_idx=SEP_TOKEN_INDEX,
    device=DEVICE
)

print(f"Reconstruction MSE: {recon_metrics['mse']:.6f}")
print(f"Explained variance: {recon_metrics['explained_variance']:.4f}")
print(f"Per-sample MSE - Mean: {recon_metrics['per_sample_mse'].mean():.6f}, Std: {recon_metrics['per_sample_mse'].std():.6f}")

# Plot MSE distribution
plt.figure(figsize=(10, 5))
plt.hist(recon_metrics['per_sample_mse'].numpy(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Per-sample MSE')
plt.ylabel('Frequency')
plt.title(f'SAE Reconstruction Error Distribution')
plt.yscale('log')
plt.grid(alpha=0.3)
if SAVE_RESULTS:
    plt.savefig(f"{SAVE_DIR}sae_recon_mse_dist.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# Identify special features (requires attention weights)
# Extract attention weights from SEP token to input positions
with torch.no_grad():
    alpha_d1_all = []
    alpha_d2_all = []
    
    for inputs, _ in tqdm(val_dl, desc="Extracting attention weights", leave=False):
        inputs = inputs.to(DEVICE)
        
        # Run model with attention cache
        logits, cache = model.run_with_cache(inputs)
        
        # Get attention scores from SEP token (position LIST_LEN) in layer 0
        attn_layer0 = cache["blocks.0.attn.hook_attn_scores"]  # [batch, n_heads, seq_len, seq_len]
        
        # SEP attends to d1 and d2 (positions 0 and 1)
        sep_pos = SEP_TOKEN_INDEX
        alpha_d1 = attn_layer0[:, :, sep_pos, 0].mean(dim=1)  # Average over heads
        alpha_d2 = attn_layer0[:, :, sep_pos, 1].mean(dim=1)
        
        alpha_d1_all.append(alpha_d1.cpu())
        alpha_d2_all.append(alpha_d2.cpu())

alpha_d1_all = torch.cat(alpha_d1_all)
alpha_d2_all = torch.cat(alpha_d2_all)

# Identify special features
special_features_info = identify_special_features(
    sae_acts_all=sae_acts_all,
    alpha_d1_all=alpha_d1_all,
    alpha_d2_all=alpha_d2_all,
    threshold=0.5
)

print(f"Special features (|corr| > 0.5): {special_features_info['n_special_features']}")
print(f"Max correlation: {special_features_info['max_correlation']:.4f}")
print(f"Mean abs correlation: {special_features_info['mean_abs_correlation']:.4f}")

if special_features_info['special_features']:
    print("\nTop special features:")
    top_special = sorted(
        special_features_info['special_features'],
        key=lambda x: abs(x['correlation']),
        reverse=True
    )[:10]
    for feat in top_special:
        print(f"  Feature {feat['feature_idx']}: {feat['type']}, corr={feat['correlation']:.4f}")

# %%
# Comprehensive feature activation analysis

# 1. Feature firing frequency
feature_firing_freq = (sae_acts_all > 0).float().mean(dim=0).numpy()
active_features = np.where(feature_firing_freq > 0)[0]
n_active = len(active_features)

print(f"Active features: {n_active} / {D_SAE}")

# Sort by firing frequency
sorted_indices = np.argsort(feature_firing_freq)[::-1]
top_n = min(30, n_active)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart of top features by firing rate
ax = axes[0]
ax.bar(range(top_n), feature_firing_freq[sorted_indices[:top_n]], color='steelblue', edgecolor='black')
ax.set_xlabel('Feature rank')
ax.set_ylabel('Firing frequency')
ax.set_title(f'Top {top_n} Features by Firing Rate')
ax.grid(alpha=0.3)

# Histogram of firing frequencies
ax = axes[1]
ax.hist(feature_firing_freq[feature_firing_freq > 0], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Firing frequency')
ax.set_ylabel('Number of features')
ax.set_title('Distribution of Feature Firing Frequencies')
ax.set_yscale('log')
ax.grid(alpha=0.3)

plt.tight_layout()
if SAVE_RESULTS:
    plt.savefig(f"{SAVE_DIR}sae_firing_frequencies.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 2. Activation strength analysis per feature

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Mean activation strength (when active)
mean_act_strength = []
for i in range(D_SAE):
    active_mask = sae_acts_all[:, i] > 0
    if active_mask.any():
        mean_act_strength.append(sae_acts_all[active_mask, i].mean().item())
    else:
        mean_act_strength.append(0)
mean_act_strength = np.array(mean_act_strength)

# Top features by mean strength
ax = axes[0, 0]
top_strength_idx = np.argsort(mean_act_strength)[::-1][:top_n]
ax.bar(range(top_n), mean_act_strength[top_strength_idx], color='coral', edgecolor='black')
ax.set_xlabel('Feature rank')
ax.set_ylabel('Mean activation (when active)')
ax.set_title(f'Top {top_n} Features by Mean Activation Strength')
ax.grid(alpha=0.3)

# Firing frequency vs mean strength scatter
ax = axes[0, 1]
active_mask = feature_firing_freq > 0
ax.scatter(feature_firing_freq[active_mask], mean_act_strength[active_mask], 
           alpha=0.6, s=50, c='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Firing frequency')
ax.set_ylabel('Mean activation strength')
ax.set_title('Firing Frequency vs Activation Strength')
ax.set_xscale('log')
ax.grid(alpha=0.3)

# Max activation per feature
max_act_strength = sae_acts_all.max(dim=0)[0].numpy()
ax = axes[1, 0]
ax.bar(range(top_n), max_act_strength[top_strength_idx], color='orange', edgecolor='black')
ax.set_xlabel('Feature rank')
ax.set_ylabel('Max activation')
ax.set_title(f'Top {top_n} Features by Max Activation')
ax.grid(alpha=0.3)

# Total activation (sum over all samples)
total_activation = sae_acts_all.sum(dim=0).numpy()
ax = axes[1, 1]
top_total_idx = np.argsort(total_activation)[::-1][:top_n]
ax.bar(range(top_n), total_activation[top_total_idx], color='mediumseagreen', edgecolor='black')
ax.set_xlabel('Feature rank')
ax.set_ylabel('Total activation')
ax.set_title(f'Top {top_n} Features by Total Activation')
ax.grid(alpha=0.3)

plt.tight_layout()
if SAVE_RESULTS:
    plt.savefig(f"{SAVE_DIR}sae_activation_strengths.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 3. Input pattern analysis for top features

def plot_feature_activation_patterns(feat_idx, d1_all, d2_all, sae_acts_all, n_digits=100, ax=None):
    """Plot which (d1, d2) pairs activate a specific feature."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Create activation matrix
    act_matrix = torch.zeros(n_digits, n_digits)
    count_matrix = torch.zeros(n_digits, n_digits)
    
    for i in range(len(d1_all)):
        d1, d2 = d1_all[i].item(), d2_all[i].item()
        act_matrix[d1, d2] += sae_acts_all[i, feat_idx].item()
        count_matrix[d1, d2] += 1
    
    # Average activation
    act_matrix = torch.where(count_matrix > 0, act_matrix / count_matrix, act_matrix)
    
    # Plot
    im = ax.imshow(act_matrix.numpy(), cmap='hot', origin='lower', aspect='auto')
    ax.set_xlabel('d2')
    ax.set_ylabel('d1')
    ax.set_title(f'Feature {feat_idx} Activations')
    plt.colorbar(im, ax=ax, label='Mean activation')
    
    # Add diagonal line (d1=d2)
    ax.plot([0, n_digits-1], [0, n_digits-1], 'c--', alpha=0.5, linewidth=1)
    
    return ax

# Plot top 6 features by total activation
top_6_features = np.argsort(total_activation)[::-1][:6]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, feat_idx in enumerate(top_6_features):
    plot_feature_activation_patterns(feat_idx, d1_all, d2_all, sae_acts_all, N_DIGITS, ax=axes[idx])

plt.tight_layout()
if SAVE_RESULTS:
    plt.savefig(f"{SAVE_DIR}sae_top_feature_patterns.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 4. Feature statistics table
print("Top 20 Features Summary:\n")
print(f"{'Feat':>4} | {'Fire%':>7} | {'MeanAct':>8} | {'MaxAct':>8} | {'TotalAct':>10} | {'Top Inputs (d1,d2)'}")
print("-" * 80)

for rank, feat_idx in enumerate(top_total_idx[:20]):
    firing_pct = feature_firing_freq[feat_idx] * 100
    mean_act = mean_act_strength[feat_idx]
    max_act = max_act_strength[feat_idx]
    total_act = total_activation[feat_idx]
    
    # Find top 3 activating inputs
    top_activations_idx = sae_acts_all[:, feat_idx].argsort(descending=True)[:3]
    top_inputs = [(d1_all[i].item(), d2_all[i].item()) for i in top_activations_idx]
    top_inputs_str = ", ".join([f"({d1},{d2})" for d1, d2 in top_inputs])
    
    print(f"{feat_idx:4d} | {firing_pct:6.2f}% | {mean_act:8.4f} | {max_act:8.4f} | {total_act:10.2f} | {top_inputs_str}")

# %%
# 5. Digit distribution analysis for each feature

def compute_feature_digit_stats(feat_idx, d1_all, d2_all, sae_acts_all, n_digits=100):
    """
    Compute digit distribution statistics for a feature.
    
    Returns dict with:
        - n_inputs: number of inputs that activate this feature
        - all_digit_dist: percentage distribution across all 2n digits
        - d1_digit_dist: percentage distribution for d1 positions only
        - d2_digit_dist: percentage distribution for d2 positions only
    """
    # Find inputs where feature activates
    active_mask = sae_acts_all[:, feat_idx] > 0
    n_inputs = active_mask.sum().item()
    
    if n_inputs == 0:
        return {
            'n_inputs': 0,
            'all_digit_dist': np.zeros(n_digits),
            'd1_digit_dist': np.zeros(n_digits),
            'd2_digit_dist': np.zeros(n_digits),
        }
    
    # Get active d1 and d2 values
    d1_active = d1_all[active_mask].numpy()
    d2_active = d2_all[active_mask].numpy()
    
    # Count digit occurrences
    d1_counts = np.bincount(d1_active, minlength=n_digits)
    d2_counts = np.bincount(d2_active, minlength=n_digits)
    all_counts = d1_counts + d2_counts
    
    # Convert to percentages
    total_digits = 2 * n_inputs
    all_digit_dist = 100 * all_counts / total_digits
    d1_digit_dist = 100 * d1_counts / n_inputs
    d2_digit_dist = 100 * d2_counts / n_inputs
    
    return {
        'n_inputs': n_inputs,
        'all_digit_dist': all_digit_dist,
        'd1_digit_dist': d1_digit_dist,
        'd2_digit_dist': d2_digit_dist,
    }

# Compute for top features
n_features_to_analyze = min(20, n_active)
top_features = top_total_idx[:n_features_to_analyze]

feature_stats = {}
for feat_idx in tqdm(top_features, desc="Computing digit stats"):
    feature_stats[feat_idx] = compute_feature_digit_stats(
        feat_idx, d1_all, d2_all, sae_acts_all, N_DIGITS
    )

# %%
# Display digit distribution table for top features
print(f"\n{'='*100}")
print(f"DIGIT DISTRIBUTION ANALYSIS FOR TOP {n_features_to_analyze} FEATURES")
print(f"{'='*100}\n")

for feat_idx in top_features[:10]:  # Show first 10
    stats = feature_stats[feat_idx]
    n_inputs = stats['n_inputs']
    
    print(f"\n{'─'*100}")
    print(f"Feature {feat_idx}: Activates on {n_inputs} inputs ({n_inputs*2} total digits)")
    print(f"{'─'*100}")
    
    # Find top digits overall
    all_dist = stats['all_digit_dist']
    top_digits_all = np.argsort(all_dist)[::-1][:10]
    
    print(f"\nTop digits (combined d1 + d2):")
    for digit in top_digits_all:
        if all_dist[digit] > 0:
            print(f"  Digit {digit:2d}: {all_dist[digit]:5.1f}%")
    
    # Find top digits for d1 and d2 separately
    d1_dist = stats['d1_digit_dist']
    d2_dist = stats['d2_digit_dist']
    top_digits_d1 = np.argsort(d1_dist)[::-1][:5]
    top_digits_d2 = np.argsort(d2_dist)[::-1][:5]
    
    print(f"\nTop d1 digits:")
    for digit in top_digits_d1:
        if d1_dist[digit] > 0:
            print(f"  Digit {digit:2d}: {d1_dist[digit]:5.1f}%")
    
    print(f"\nTop d2 digits:")
    for digit in top_digits_d2:
        if d2_dist[digit] > 0:
            print(f"  Digit {digit:2d}: {d2_dist[digit]:5.1f}%")

# %%
# Visualize digit distributions for top features
n_vis = min(6, len(top_features))
fig, axes = plt.subplots(n_vis, 3, figsize=(18, 4*n_vis))
if n_vis == 1:
    axes = axes[np.newaxis, :]

for idx, feat_idx in enumerate(top_features[:n_vis]):
    stats = feature_stats[feat_idx]
    
    # Plot combined distribution
    ax = axes[idx, 0]
    digits = np.arange(N_DIGITS)
    ax.bar(digits, stats['all_digit_dist'], color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Feature {feat_idx}: All digits (n={stats["n_inputs"]} inputs)')
    ax.grid(alpha=0.3, axis='y')
    ax.set_xlim(-1, N_DIGITS)
    
    # Plot d1 distribution
    ax = axes[idx, 1]
    ax.bar(digits, stats['d1_digit_dist'], color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Feature {feat_idx}: d1 positions only')
    ax.grid(alpha=0.3, axis='y')
    ax.set_xlim(-1, N_DIGITS)
    
    # Plot d2 distribution
    ax = axes[idx, 2]
    ax.bar(digits, stats['d2_digit_dist'], color='mediumseagreen', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Feature {feat_idx}: d2 positions only')
    ax.grid(alpha=0.3, axis='y')
    ax.set_xlim(-1, N_DIGITS)

plt.tight_layout()
if SAVE_RESULTS:
    plt.savefig(f"{SAVE_DIR}sae_digit_distributions.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# Create comprehensive dataframe for all features
rows = []
for feat_idx in top_features:
    stats = feature_stats[feat_idx]
    
    # Get top 5 digits for each category
    all_dist = stats['all_digit_dist']
    d1_dist = stats['d1_digit_dist']
    d2_dist = stats['d2_digit_dist']
    
    top5_all = np.argsort(all_dist)[::-1][:5]
    top5_d1 = np.argsort(d1_dist)[::-1][:5]
    top5_d2 = np.argsort(d2_dist)[::-1][:5]
    
    # Format as strings
    top_all_str = ", ".join([f"{d}({all_dist[d]:.1f}%)" for d in top5_all if all_dist[d] > 0])
    top_d1_str = ", ".join([f"{d}({d1_dist[d]:.1f}%)" for d in top5_d1 if d1_dist[d] > 0])
    top_d2_str = ", ".join([f"{d}({d2_dist[d]:.1f}%)" for d in top5_d2 if d2_dist[d] > 0])
    
    rows.append({
        'Feature': feat_idx,
        'N_Inputs': stats['n_inputs'],
        'Fire_Rate_%': feature_firing_freq[feat_idx] * 100,
        'Top_All_Digits': top_all_str,
        'Top_D1_Digits': top_d1_str,
        'Top_D2_Digits': top_d2_str,
    })

df_digit_stats = pd.DataFrame(rows)
print(f"\n{'='*120}")
print("COMPREHENSIVE DIGIT DISTRIBUTION TABLE")
print(f"{'='*120}\n")
print(df_digit_stats.to_string(index=False))

if SAVE_RESULTS:
    df_digit_stats.to_csv(f"{SAVE_DIR}sae_digit_distributions.csv", index=False)
    print(f"\nSaved to {SAVE_DIR}sae_digit_distributions.csv")

# %%
