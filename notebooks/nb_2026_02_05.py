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
# looking at:
SAE_NAME = "sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt" # MSE: 0.0042, Recon Acc: 0.8688 (old - not as high accuracy as below)
# SAE_NAME = "sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d.pt" # MSE: 0.0036, Recon Acc: 0.8752
# SAE_NAME = "sae_d100_k4_50ksteps_2layer_100dig_64d.pt" # (classic - best recon accuracy of 0.8885) 


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

# %%
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
SPECIAL_FEAT_IDX = 30

results = feature_steering_experiment(
    model=model, sae=sae, act_mean=act_mean,
    feature_idx=SPECIAL_FEAT_IDX,
    d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
    dataset=all_ds,
    sep_idx=SEP_TOKEN_INDEX,
    n_digits=N_DIGITS,
    plot=True,
    save_dir=SAVE_DIR
)

# %% [markdown]
# the above graph just shows 5 random input examples
# 
# TODO
# 1) need to check all other logits - it may look good here but maybe another logit overtakes
# 2) try 'non-special' features - does the same pattern appear (yes) ==> why is special one 'special'? have i just found a vis to fit my hopes?
# 3) try actually steering with it --> should be able to predict that we can boost/damp f30 by x% and swap the ouputs logits ==> (maybe) we've got a feature that controls output order
# 
# GOAL - show that this feature reliably swaps outputs (or it doesnt!)
# 

# %%
# let's focus on the logit at o_1 for a particular input example (91, 83).
# plot red line for d2=83 logit, blue line for d1=91 logit, and grey lines for all other logits
# ==> should be 100 lines in total! x axis (feature 30 scale) should go from -1 to 2 

input_example_idx = 0 # get idx for (91,83) in the dataset all_ds