#%% [markdown]


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
from nb_utils import setup_notebook, load_transformer_model, load_sae

# Import project utilities
sys.path.insert(0, '..')
from model_scripts.model_utils import build_attention_mask, parse_model_name_safe, configure_runtime, load_model
from model_scripts.data import get_dataset
from model_scripts.sae_analysis import (
    collect_sae_activations,
    create_feature_heatmaps,
    compute_reconstruction_metrics,
    identify_special_features,
    load_sae_from_wandb_run,
    load_sae_from_local,
    compare_sweep_runs,
)

# Setup device and seeds
DEVICE = setup_notebook(seed=42)

# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
SAVE_NAME = 'sae_d100_k4_50ksteps_2layer_100dig_64d.pt'

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
sae, sae_cfg = load_sae(SAVE_NAME, D_MODEL, device=DEVICE)
D_SAE = sae_cfg['dict_size']
TOP_K = sae_cfg['k']

# Load activation mean from checkpoint (for centering)
SAE_PATH = os.path.join('../results/sae_models', SAVE_NAME)
sae_checkpoint = torch.load(SAE_PATH, map_location=DEVICE, weights_only=False)
act_mean = sae_checkpoint["act_mean"].to(DEVICE)

# %%
# Example usage for loading SAEs from W&B sweeps

# Load a specific run by ID
sae_data = load_sae_from_wandb_run("nqie9jok", device=DEVICE)
sae = sae_data["sae"]
act_mean = sae_data["act_mean"]

# Or load from local file
# sae_data = load_sae_from_local("../results/sae_models/sweep_runs/sae_d256_k4_lr0.001_seed42_2layer_100dig_64d.pt", device=DEVICE)

# Compare all runs in sweep
df = compare_sweep_runs()
print(df.head())
