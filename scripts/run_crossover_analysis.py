"""
Crossover Analysis Script

Runs the crossover analysis pipeline on all inputs and saves results to disk.
Designed to run on GPU via SLURM job submission.
"""
# %%
import os
import sys
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import project utilities
from src.utils.nb_utils import setup_notebook, load_transformer_model, load_sae
from src.data.datasets import get_dataset
from src.sae import *  # Import all SAE analysis utilities

# Configuration
MODEL_NAME = '2layer_100dig_64d'
SAE_NAME = "sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt"
SPECIAL_FEAT_IDX = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = Path("results/xover")
BATCH_SIZE = 64  # Batching for efficiency

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# %%
print("="*60)
print("CROSSOVER ANALYSIS - GPU JOB")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"SAE: {SAE_NAME}")
print(f"Feature: {SPECIAL_FEAT_IDX}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Results directory: {RESULTS_DIR}")
print("="*60)

# Setup
_ = setup_notebook(seed=42)

# Load models
print("\n[1/6] Loading models...")
model, model_cfg = load_transformer_model(MODEL_NAME, device=DEVICE)
D_MODEL = model_cfg['d_model']
N_DIGITS = model_cfg['n_digits']
LIST_LEN = model_cfg['list_len']
SEP_TOKEN_INDEX = model_cfg['sep_token_index']

sae, sae_cfg = load_sae(SAE_NAME, D_MODEL, device=DEVICE)

# Load activation mean
SAE_PATH = os.path.join('results/sae_models', SAE_NAME)
sae_checkpoint = torch.load(SAE_PATH, map_location=DEVICE, weights_only=False)
act_mean = sae_checkpoint["act_mean"].to(DEVICE)

# Load dataset
print("\n[2/6] Loading dataset...")
train_ds, val_ds = get_dataset(
    n_digits=N_DIGITS,
    list_len=LIST_LEN,
    no_dupes=False,
    train_dupes_only=False
)
all_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
print(f"Total inputs: {len(all_ds)}")

# Collect SAE activations
print("\n[3/6] Collecting SAE activations...")
from torch.utils.data import DataLoader
all_dl = DataLoader(all_ds, batch_size=128, shuffle=False)

d1_all, d2_all, sae_acts_all = collect_sae_activations(
    model=model,
    sae=sae,
    val_dl=all_dl,
    act_mean=act_mean,
    layer_idx=0,
    sep_idx=SEP_TOKEN_INDEX,
    device=DEVICE
)

# Find crossovers
print(f"\n[4/6] Finding crossovers (feature {SPECIAL_FEAT_IDX})...")
xovers_df = get_xovers_df(
    model=model, sae=sae, act_mean=act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, d2_all=d2_all,
    sae_acts_all=sae_acts_all,
    dataset=all_ds,
    layer_idx=0, sep_idx=SEP_TOKEN_INDEX, n_digits=N_DIGITS,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

# Save crossovers
xovers_path = RESULTS_DIR / f"xovers_feat{SPECIAL_FEAT_IDX}.csv"
xovers_df.to_csv(xovers_path, index=False)
print(f"✓ Saved crossovers to {xovers_path}")

# Quick stats
print(f"\nCrossover statistics:")
print(f"  Total inputs: {len(xovers_df)}")
print(f"  Inputs with feature firing: {(xovers_df['feat_orig'] > 0).sum()}")
print(f"  Inputs with crossovers: {((xovers_df['n_o1_xover'] > 0) | (xovers_df['n_o2_xover'] > 0)).sum()}")

# Get swap bounds
print(f"\n[5/6] Identifying swap zones...")
swap_bounds_df = get_output_swap_bounds(xovers_df)

# Save swap bounds
swap_bounds_path = RESULTS_DIR / f"swap_bounds_feat{SPECIAL_FEAT_IDX}.csv"
swap_bounds_df.to_csv(swap_bounds_path, index=False)
print(f"✓ Saved swap bounds to {swap_bounds_path}")

# Quick stats
valid_swaps = swap_bounds_df['failure_reason'].isna().sum()
print(f"\nSwap bounds statistics:")
print(f"  Valid swap zones: {valid_swaps}")
print(f"  Failed: {len(swap_bounds_df) - valid_swaps}")

# Verify swaps
print(f"\n[6/6] Verifying output swaps...")
swap_results_df = swap_outputs(
    model=model, sae=sae, act_mean=act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    swap_bounds_df=swap_bounds_df,
    d1_all=d1_all, d2_all=d2_all,
    sae_acts_all=sae_acts_all,
    dataset=all_ds,
    layer_idx=0, sep_idx=SEP_TOKEN_INDEX, n_digits=N_DIGITS,
    device=DEVICE
)

# Save swap results
swap_results_path = RESULTS_DIR / f"swap_results_feat{SPECIAL_FEAT_IDX}.csv"
swap_results_df.to_csv(swap_results_path, index=False)
print(f"✓ Saved swap results to {swap_results_path}")

# Final stats
total = len(swap_results_df)
swapped = swap_results_df['swapped'].sum()
print(f"\nSwap verification results:")
print(f"  Total verified: {total}")
print(f"  Successfully swapped: {swapped} ({swapped/total*100:.1f}%)")
print(f"  Failed to swap: {total - swapped} ({(total-swapped)/total*100:.1f}%)")

print("\n" + "="*60)
print("CROSSOVER ANALYSIS COMPLETE")
print("="*60)
print(f"\nResults saved to {RESULTS_DIR}:")
print(f"  - {xovers_path.name}")
print(f"  - {swap_bounds_path.name}")
print(f"  - {swap_results_path.name}")
