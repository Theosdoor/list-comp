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
test_egs = [(75,29)]

results = feature_steering_experiment(
    model, sae, act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, 
    d2_all=d2_all, 
    sae_acts_all=sae_acts_all, 
    dataset=all_ds,
    scale_range=[-10.0, 10.0],
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
# some of these the model gets wrong. Lets filter the ones the model gets wrong originally, and see the swap bounds for those

# Get original (unpatched) predictions for ALL inputs via a direct forward pass
# Use targets from the dataloader — same as the accuracy() function used during training
from src.models.utils import accuracy as model_accuracy

all_d1, all_d2, all_o1_pred, all_o2_pred, all_o1_tgt, all_o2_tgt = [], [], [], [], [], []

model.eval()
with torch.no_grad():
    for inputs, targets in all_dl:
        inputs = inputs.to(DEVICE)
        tgts = targets[:, LIST_LEN + 1:].to(DEVICE)   # same slice as accuracy()
        logits = model(inputs)[:, LIST_LEN + 1:]
        preds = logits.argmax(dim=-1)                  # [batch, LIST_LEN]
        all_d1.extend(inputs[:, 0].cpu().tolist())
        all_d2.extend(inputs[:, 1].cpu().tolist())
        all_o1_pred.extend(preds[:, 0].cpu().tolist())
        all_o2_pred.extend(preds[:, 1].cpu().tolist())
        all_o1_tgt.extend(tgts[:, 0].cpu().tolist())
        all_o2_tgt.extend(tgts[:, 1].cpu().tolist())

orig_preds_df = pd.DataFrame({
    'd1': all_d1, 'd2': all_d2,
    'orig_o1': all_o1_pred, 'orig_o2': all_o2_pred,
    'tgt_o1': all_o1_tgt,   'tgt_o2': all_o2_tgt,
})
orig_preds_df['o1_correct'] = orig_preds_df['orig_o1'] == orig_preds_df['tgt_o1']
orig_preds_df['o2_correct'] = orig_preds_df['orig_o2'] == orig_preds_df['tgt_o2']
orig_preds_df['token_acc'] = (orig_preds_df['o1_correct'].astype(float) + orig_preds_df['o2_correct'].astype(float)) / 2

# Per-token accuracy on val — should match model_accuracy() = 0.9145
n_val = len(val_ds)
val_preds_df = orig_preds_df.iloc[-n_val:]
print(f"Val per-token accuracy: {val_preds_df['token_acc'].mean():.4f}  (reference: 0.9145)")
print()

n_total = len(orig_preds_df)
print(f"All inputs: {n_total}")
print(f"  Per-token accuracy: {orig_preds_df['token_acc'].mean():.4f}")
print(f"  Both correct:  {(orig_preds_df['token_acc'] == 1.0).sum()} ({(orig_preds_df['token_acc'] == 1.0).mean()*100:.1f}%)")
print(f"  Partial (1/2): {(orig_preds_df['token_acc'] == 0.5).sum()} ({(orig_preds_df['token_acc'] == 0.5).mean()*100:.1f}%)")
print(f"  Both wrong:    {(orig_preds_df['token_acc'] == 0.0).sum()} ({(orig_preds_df['token_acc'] == 0.0).mean()*100:.1f}%)")

# Merge into swap_bounds — use 3-way correctness label for breakdown
def _correctness_label(row):
    if row['o1_correct'] and row['o2_correct']:
        return 'both_correct'
    elif row['o1_correct'] or row['o2_correct']:
        return 'partial'
    else:
        return 'both_wrong'

orig_preds_df['correctness'] = orig_preds_df.apply(_correctness_label, axis=1)

# Merge correctness into swap_bounds (covers ALL rows incl. failures)
swap_bounds_annotated = swap_bounds_df.merge(
    orig_preds_df[['d1', 'd2', 'orig_o1', 'orig_o2', 'o1_correct', 'o2_correct', 'token_acc', 'correctness']],
    on=['d1', 'd2'], how='left'
).merge(
    swap_results_df[['d1', 'd2', 'swapped']],
    on=['d1', 'd2'], how='left'
)

wrong_bounds = swap_bounds_annotated[swap_bounds_annotated['token_acc'] < 1.0].copy()
valid_wrong = wrong_bounds['failure_reason'].isna()

print(f"\nSwap bounds for inputs with at least one wrong token: {len(wrong_bounds)}")
print(f"  Valid swap zones: {valid_wrong.sum()}")
print(f"  Successfully swapped: {wrong_bounds.loc[valid_wrong, 'swapped'].sum()}")
print(f"  Swap zone widths (valid):")
print(wrong_bounds.loc[valid_wrong, 'swap_zone_width'].describe())
display(wrong_bounds)

# %%
# Failure reason breakdown by whether model originally got the input correct

# Fill NaN failure_reason with 'success' for readability
annotated = swap_bounds_annotated.copy()
annotated['failure_reason'] = annotated['failure_reason'].fillna('success')
# correctness column already set: 'both_correct', 'partial', 'both_wrong'

breakdown = (
    annotated
    .groupby(['failure_reason', 'correctness'])
    .size()
    .unstack(fill_value=0)
)
# Ensure consistent column order
for col in ['both_correct', 'partial', 'both_wrong']:
    if col not in breakdown.columns:
        breakdown[col] = 0
breakdown = breakdown[['both_correct', 'partial', 'both_wrong']]
breakdown['all'] = breakdown.sum(axis=1)
breakdown.loc['TOTAL'] = breakdown.sum()

print("Failure reason breakdown by original model correctness:")
display(breakdown)
# %%
