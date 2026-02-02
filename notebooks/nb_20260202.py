# %%
# SETUP

import os

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

# Use BatchTopKSAE from dictionary_learning library
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

# Import project utilities (add parent to path for imports)
import sys
sys.path.insert(0, '..')
from model_scripts.model_utils import configure_runtime, load_model, parse_model_name_safe
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

# Set Device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")
torch.set_grad_enabled(False) # don't need gradients - analysis only

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Configuration (Must match training) ---
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)
SAVE_FOLDER = '../results/sae_models'

SAVE_NAME = 'sae_d100_k4_50ksteps_2layer_100dig_64d.pt'
SAE_PATH = os.path.join(SAVE_FOLDER, SAVE_NAME)

# from model
D_MODEL = MODEL_CFG.d_model
N_LAYERS = MODEL_CFG.n_layers
N_HEADS = 1
LIST_LEN = 2
N_DIGITS = MODEL_CFG.n_digits
SEP_TOKEN_INDEX = 2  # Position of SEP in [d1, d2, SEP, o1, o2]

# Output Config
SAVE_RESULTS = False
if SAVE_RESULTS:
    SAVE_DIR = "../results/sae_results/"
else:
    SAVE_DIR = None

# --- Load Models ---
MODEL_PATH = "../models/" + MODEL_NAME + ".pt"

# Setup Runtime (required by model_utils)
configure_runtime(
    list_len=LIST_LEN,
    seq_len=2 * LIST_LEN + 1,  # [d1, d2, SEP, o1, o2] = 5
    vocab=N_DIGITS + 2,  # digits + MASK + SEP
    device=DEVICE
)

# Load base transformer model (with required kwargs)
try:
    model = load_model(
        MODEL_PATH,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        ln=False,
        use_bias=False,
        use_wv=False,
        use_wo=False
    )
    print(f"✓ Loaded base model from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    raise

# Load SAE using library's BatchTopKSAE
sae_checkpoint = torch.load(SAE_PATH, map_location=DEVICE, weights_only=False)

# Extract config from checkpoint
sae_cfg = sae_checkpoint.get("cfg", {})
D_SAE = sae_cfg.get("dict_size", sae_cfg.get("d_sae", 256))
TOP_K = sae_cfg.get("k", 4)

# Library's BatchTopKSAE uses (activation_dim, dict_size, k) constructor
sae = BatchTopKSAE(
    activation_dim=D_MODEL,
    dict_size=D_SAE,
    k=TOP_K
).to(DEVICE)

# Load state dict (handles both old and new formats)
old_state_dict = sae_checkpoint["state_dict"]
if "W_enc" in old_state_dict:
    # Legacy format conversion (for old sae.pt)
    new_state_dict = {
        "encoder.weight": old_state_dict["W_enc"].T,
        "encoder.bias": old_state_dict["b_enc"],
        "decoder.weight": old_state_dict["W_dec"].T,
        "b_dec": old_state_dict["b_dec"],
    }
    sae.load_state_dict(new_state_dict, strict=False)
    print("  (Converted legacy checkpoint format)")
else:
    sae.load_state_dict(old_state_dict)
    print(f"  (Threshold: {sae.threshold.item():.4f})")

# Load the mean for centering (critical for SAE)
act_mean = sae_checkpoint["act_mean"].to(DEVICE)

print(f"✓ Loaded SAE from {SAE_PATH}")
print(f"  - Latent dim: {D_SAE}")
print(f"  - TopK: {TOP_K}")

# %%
# Example usage for loading SAEs from W&B sweeps

# Load a specific run by ID
# sae_data = load_sae_from_wandb_run("nqie9jok", device=device)
# sae = sae_data["sae"]
# act_mean = sae_data["act_mean"]

# Or load from local file
# sae_data = load_sae_from_local("../results/sae_models/sweep_runs/sae_d256_k4_lr0.001_seed42_2layer_100dig_64d.pt", device=device)

# Compare all runs in sweep
# df = compare_sweep_runs()
# print(df.head())

# %%

