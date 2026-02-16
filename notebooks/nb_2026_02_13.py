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

# Perform crossover analysis with bisection for exact values (using new refactored function)
from src.sae.steering import analyze_feature_crossovers

crossover_df = analyze_feature_crossovers(
    results=coarse_results,
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
# What's the accuracy when special feature is ablated?
def evaluate_with_ablated_feature(
    model, sae, act_mean, feature_idx, 
    dataloader, layer_idx, sep_idx, n_digits, device
):
    """
    Evaluate model accuracy when a specific SAE feature is ablated.
    
    Args:
        model: The transformer model to evaluate
        sae: Sparse autoencoder for feature extraction
        act_mean: Mean activation for centering
        feature_idx: Index of the feature to ablate
        dataloader: DataLoader containing (inputs, targets) batches
        layer_idx: Layer index to apply the ablation hook
        sep_idx: Position of the separator token
        n_digits: Number of digit tokens in vocabulary
        device: torch device
        
    Returns:
        tuple: (accuracy, num_correct, num_total)
    """
    model.eval()
    sae.eval()
    
    correct = 0
    total = 0
    
    def ablation_hook(module, input, output):
        """Hook to ablate specific feature at SEP token position."""
        # Extract activations at SEP token: [batch, d_model]
        acts = output[:, sep_idx, :].clone()
        
        # Center, encode, ablate, decode, and un-center
        acts_centered = acts - act_mean
        z = sae.encode(acts_centered)
        z[:, feature_idx] = 0  # Ablate the feature
        acts_reconstructed = sae.decode(z) + act_mean
        
        # Replace SEP token activations
        output[:, sep_idx, :] = acts_reconstructed
        return output
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            
            # Register hook, run forward pass, then remove hook
            hook_handle = model.blocks[layer_idx].register_forward_hook(ablation_hook)
            
            try:
                logits = model(inputs)  # [batch, seq, vocab]
                
                # Get predictions for last two positions
                preds_o1 = logits[:, -2, :n_digits].argmax(dim=-1)
                preds_o2 = logits[:, -1, :n_digits].argmax(dim=-1)
                
                # Get ground truth from targets
                targets_o1 = targets[:, -2]
                targets_o2 = targets[:, -1]
                
                # Count correct predictions (both positions must match)
                correct_both = (preds_o1 == targets_o1) & (preds_o2 == targets_o2)
                correct += correct_both.sum().item()
                total += inputs.shape[0]
                
            finally:
                hook_handle.remove()
    
    accuracy = correct / total
    return accuracy, correct, total


def evaluate_baseline(model, dataloader, n_digits, device):
    """Evaluate baseline model accuracy without any ablation."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
            
            # Get predictions and targets for last two positions
            preds_o1 = logits[:, -2, :n_digits].argmax(dim=-1)
            preds_o2 = logits[:, -1, :n_digits].argmax(dim=-1)
            targets_o1 = targets[:, -2]
            targets_o2 = targets[:, -1]
            
            # Count correct predictions
            correct_both = (preds_o1 == targets_o1) & (preds_o2 == targets_o2)
            correct += correct_both.sum().item()
            total += inputs.shape[0]
    
    return correct / total, correct, total


val_acc, val_correct, val_total = evaluate_with_ablated_feature(
    model=model,
    sae=sae,
    act_mean=act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    dataloader=val_dl,
    layer_idx=0,
    sep_idx=SEP_TOKEN_INDEX,
    n_digits=N_DIGITS,
    device=DEVICE
)
print(f"\nValidation Results (Feature {SPECIAL_FEAT_IDX} Ablated):")
print(f"  Correct: {val_correct}/{val_total}")
print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")


baseline_acc, baseline_correct, baseline_total = evaluate_baseline(
    model=model,
    dataloader=val_dl,
    n_digits=N_DIGITS,
    device=DEVICE
)
print(f"\nBaseline Results:")
print(f"  Correct: {baseline_correct}/{baseline_total}")
print(f"  Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")

# %%
# Ablate all features and visualize impact

print("\nEvaluating all features (0-99)...")
results = []

for feature_idx in tqdm(range(100)):
    val_acc, _, _ = evaluate_with_ablated_feature(
        model=model,
        sae=sae,
        act_mean=act_mean,
        feature_idx=feature_idx,
        dataloader=val_dl,
        layer_idx=0,
        sep_idx=SEP_TOKEN_INDEX,
        n_digits=N_DIGITS,
        device=DEVICE
    )
    results.append({"feature_idx": feature_idx, "accuracy": val_acc})

# Create DataFrame and sort by accuracy (lowest = most impactful)
df_accuracies = pd.DataFrame(results).sort_values("accuracy").reset_index(drop=True)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_accuracies, x="feature_idx", y="accuracy", color="steelblue")
plt.xlabel("Feature Index")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy by Ablated Feature (Sorted by Impact)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Display most impactful features
print("\nTop 10 Most Impactful Features (Lowest Accuracy When Ablated):")
print(df_accuracies.head(10).to_string(index=False))

# %% [markdown]
# Okay - so F30 reduces val accuracy to 27%, 
# and the second worst to ablate is F54 which reduces val acc to 70% --> all other features hang around 70% !
# ==> F30 is defo carrying important info - but why are all the others 'equally' important? 

# %%
# CROSSOVER ANALYSIS: Load pre-computed results from GPU job
import ast

# Helper to parse list columns from CSV
def parse_list_column(df, col_name):
    """Parse string representation of lists back to actual lists."""
    df[col_name] = df[col_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

# Load crossovers
xovers_df = pd.read_csv(f'../results/xover/xovers_feat{SPECIAL_FEAT_IDX}.csv')
parse_list_column(xovers_df, 'o1_crossovers')
parse_list_column(xovers_df, 'o2_crossovers')

# Analyze crossover statistics
print(f"\nTotal inputs: {len(xovers_df)}")
print(f"Inputs with feature firing: {(xovers_df['feat_orig'] > 0).sum()}")
print(f"Inputs with no crossovers: {((xovers_df['n_o1_xover'] == 0) & (xovers_df['n_o2_xover'] == 0)).sum()}")
print(f"\nCrossover pattern distribution:")
print(xovers_df.groupby(['n_o1_xover', 'n_o2_xover']).size().to_frame('count'))

# Display sample of crossovers
print(f"\nSample of inputs with crossovers:")
display(xovers_df[xovers_df['n_o1_xover'] > 0].head(10))

# %% [markdown]
# 7054 have valid crossovers, and F30 firest on 7054 / 10k inputs 
# - presumably these are the same set of inputs

# %%
# SWAP BOUNDS: Load pre-computed swap zones
swap_bounds_df = pd.read_csv(f'../results/xover/swap_bounds_feat{SPECIAL_FEAT_IDX}.csv')

print(f"\nTotal inputs processed: {len(swap_bounds_df)}")
print(f"Valid swap zones found: {swap_bounds_df['failure_reason'].isna().sum()}")
print(f"\nFailure reason breakdown:")
print(swap_bounds_df['failure_reason'].value_counts())

# Display successful swap zones
valid_swaps = swap_bounds_df[swap_bounds_df['failure_reason'].isna()]
print(f"\nSwap zone statistics:")
print(f"Mean swap zone width: {valid_swaps['swap_zone_width'].mean():.3f}")
print(f"Median swap zone width: {valid_swaps['swap_zone_width'].median():.3f}")

print(f"\nSample of valid swap zones:")
display(valid_swaps.head(10))

invalid_swaps = swap_bounds_df[swap_bounds_df['lower_bound'].isna()]

# %% [markdown]
# so out of the 7054 inputs with xover, theres only 5700 valid swap zones? seems weird

# %%
# VERIFY SWAPS: Load pre-computed swap verification results
swap_results_df = pd.read_csv(f'../results/xover/swap_results_feat{SPECIAL_FEAT_IDX}.csv')

# Analysis
total = len(swap_results_df)
swapped = swap_results_df['swapped'].sum()
print(f"\nSwap verification results:")
print(f"Total verified: {total}")
print(f"Successfully swapped: {swapped} ({swapped/total*100:.1f}%)")
print(f"Failed to swap: {total - swapped} ({(total-swapped)/total*100:.1f}%)")

# Show some examples
print(f"\nSuccessfully swapped examples:")
display(swap_results_df[swap_results_df['swapped']].head(5))

print(f"\nFailed to swap examples:")
display(swap_results_df[~swap_results_df['swapped']].head(5))

# %%
invalid_swaps

# %%
# test specific bad examples
# test_egs = [(93, 99), (38, 78)]
test_egs = [(60,44), (75,32), (9,81)]

results = feature_steering_experiment(
    model, sae, act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, 
    d2_all=d2_all, 
    sae_acts_all=sae_acts_all, 
    dataset=all_ds,
    scale_range=[-1.0, 4.0],
    test_pairs=test_egs
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
# Inspect outputs at custom scales (e.g., 2x feature activation)

# Test the first example (93, 99) at different scales
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
    custom_scales = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
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



# %%
