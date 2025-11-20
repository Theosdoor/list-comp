# %% [markdown]
# # SAE Analysis: Order-By-Scale SEP Token
# This notebook investigates the features learned by the BatchTopK SAE trained on the SEP token residual stream.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import einops

# Import project utilities
from model_utils import make_model, configure_runtime, load_model
from data import get_dataset

# Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# %% [markdown]
# ## 1. Define Classes & Load Models
# Re-defining the SAE class here to ensure safe loading without pickling issues.

# %%
# --- Configuration (Must match training) ---
class SAEConfig:
    d_model = 64
    d_sae = 64 * 32
    k = 32
    
    # Model Config
    n_layers = 2
    n_heads = 1
    list_len = 2
    n_digits = 100
    sep_token_index = 2 

# --- BatchTopK SAE Definition ---
class BatchTopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))

    def encode(self, x):
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        post_relu = F.relu(pre_acts)
        topk_values, topk_indices = torch.topk(post_relu, k=self.cfg.k, dim=-1)
        z = torch.zeros_like(post_relu)
        z.scatter_(-1, topk_indices, topk_values)
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_reconstruct = self.decode(z)
        return x_reconstruct, z

# %% [markdown]
# ### Load Base Model and SAE

# %%
# 1. Load Base Model
# Assuming you have a base model trained. If not, it will init random (less useful analysis)
MODEL_PATH = "models/2layer_100dig_64d.pt" # Update this path if needed

# Setup Runtime
configure_runtime(
    list_len=2, 
    seq_len=5, 
    vocab=102, 
    device=device
)

try:
    model = load_model(MODEL_PATH)
    print("Loaded trained base model.")
except:
    print("Warning: Loading random base model (Analysis will be noise).")
    model = make_model(
        n_layers=2, n_heads=1, d_model=64,
        ln=False, use_bias=False, freeze_wv=True, freeze_wo=True
    )

# 2. Load SAE
sae_path = "sep_token_sae_batch_topk.pt"
sae_checkpoint = torch.load(sae_path, map_location=device)

sae = BatchTopKSAE(SAEConfig()).to(device)
sae.load_state_dict(sae_checkpoint["state_dict"])

# Critical: Load the mean for centering
act_mean = sae_checkpoint["act_mean"].to(device)

print("SAE and Mean loaded successfully.")

# %% [markdown]
# ## 2. Generate Analysis Data
# We run a large batch of data through the model, capture SEP activations, and run them through the SAE.

# %%
# Generate a large validation set
# We use no_dupes=False to see full grid of interactions
val_ds, _ = get_dataset(
    list_len=2, n_digits=100, 
    train_split=1.0, # Hack to get all data
    no_dupes=False 
)
val_dl = DataLoader(val_ds, batch_size=2048, shuffle=False)

# Store results
cache_d1 = []
cache_d2 = []
cache_sae_acts = []
cache_reconst_err = []

print("Running inference...")

layer_idx = 0 # The layer we trained on
hook_name = f"blocks.{layer_idx}.hook_resid_post"

with torch.no_grad():
    for inputs, targets in tqdm(val_dl):
        inputs = inputs.to(device)
        
        # Get d1 and d2 values for analysis plotting
        d1 = inputs[:, 0]
        d2 = inputs[:, 1]
        
        # Run Base Model
        _, cache = model.run_with_cache(inputs, stop_at_layer=layer_idx+1, names_filter=hook_name)
        sep_acts = cache[hook_name][:, 2, :] # [Batch, d_model]
        
        # SAE Forward
        # Don't forget to center!
        sep_acts_centered = sep_acts - act_mean
        x_reconstruct, z = sae(sep_acts_centered)
        
        # Calculate error
        # Note: Add mean back for true reconstruction comparison, though diff is same
        err = (sep_acts - (x_reconstruct + act_mean)).pow(2).sum(dim=-1).sqrt()
        
        cache_d1.append(d1.cpu())
        cache_d2.append(d2.cpu())
        cache_sae_acts.append(z.cpu())
        cache_reconst_err.append(err.cpu())

# Concatenate
d1_all = torch.cat(cache_d1)
d2_all = torch.cat(cache_d2)
sae_acts_all = torch.cat(cache_sae_acts)
reconst_err_all = torch.cat(cache_reconst_err)

print(f"Processed {len(d1_all)} samples.")

# %% [markdown]
# ## 3. Global SAE Metrics

# %%
# 1. Explained Variance (R2)
# Total variance of the data (centered)
total_variance = (sae_acts_all - sae_acts_all.mean(0)).pow(2).sum() 
# Unexplained variance (Reconstruction error)
unexplained_variance = reconst_err_all.pow(2).sum()
r2 = 1 - (unexplained_variance / total_variance)

# 2. L0 Sparsity (Average number of firing features)
l0 = (sae_acts_all > 0).float().sum(dim=1).mean()

print(f"--- SAE Metrics ---")
print(f"Explained Variance (R^2): {r2:.4f}")
print(f"Average L0 (Sparsity):    {l0:.2f}")
print(f"Dead Features:            {(sae_acts_all.sum(0) == 0).sum().item()} / {sae.cfg.d_sae}")

# %% [markdown]
# ## 4. Feature Dashboard
# This function creates a comprehensive view of a single SAE feature.

# %%
def analyze_feature(feature_idx):
    """
    Plots a heatmap of Feature Activation over the (d1, d2) grid.
    Also shows top activating examples and Decoder weights.
    """
    acts = sae_acts_all[:, feature_idx].numpy()
    
    if acts.max() == 0:
        print(f"Feature {feature_idx} is Dead.")
        return

    # --- 1. Heatmap over d1/d2 grid ---
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'd1': d1_all.numpy(),
        'd2': d2_all.numpy(),
        'activation': acts
    })
    
    # Aggregate max activation for duplicate d1/d2 pairs (if any)
    pivot_table = df.pivot_table(index='d1', columns='d2', values='activation', aggfunc='max')
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    sns.heatmap(pivot_table, cmap="viridis", ax=axes[0])
    axes[0].invert_yaxis() # Standard cartesian
    axes[0].set_title(f"Feature {feature_idx} Activation\n(X=d2, Y=d1)")
    
    # --- 2. Histogram of Activations ---
    axes[1].hist(acts[acts > 0], bins=50)
    axes[1].set_title("Activation Histogram (non-zero)")
    axes[1].set_xlabel("Activation Strength")
    
    # --- 3. Decoder Weight Logit Lens ---
    # What token does this feature write to?
    # We project the decoder weight into the vocabulary
    decoder_dir = sae.W_dec[feature_idx].detach() # [d_model]
    
    # Get unembedding matrix from base model
    W_U = model.unembed.W_U # [d_model, vocab]
    
    logits = decoder_dir @ W_U
    top_vals, top_idxs = torch.topk(logits, k=10)
    
    # Create bar plot for logits
    tokens = [str(i.item()) if i < 100 else ("MASK" if i==100 else "SEP") for i in top_idxs]
    
    axes[2].barh(tokens[::-1], top_vals.cpu().numpy()[::-1])
    axes[2].set_title("Decoder Weight -> Logit Lens")
    axes[2].set_xlabel("Logit Contribution")
    
    plt.tight_layout()
    plt.show()
    
    # Stats
    print(f"Max Activation: {acts.max():.4f}")
    print(f"Frac Active:    {(acts > 0).mean():.4%}")

# %% [markdown]
# ## 5. Explore Top Features
# Let's look at the most frequently active features first.

# %%
# Get indices of features sorted by how often they fire
firing_rate = (sae_acts_all > 0).float().mean(dim=0)
top_features = torch.argsort(firing_rate, descending=True)[:5]

print("Analyzing Top 5 most frequent features:")
for idx in top_features:
    analyze_feature(idx.item())

# %% [markdown]
# ## 6. Search for Specific Features
# Can we find features representing concepts mentioned in the paper?
#
# 1. **d1 features:** Vertical stripes in the heatmap.
# 2. **d2 features:** Horizontal stripes in the heatmap.
# 3. **Ordering features:** Triangles (active only when d1 > d2 or vice versa).

# %%
# Calculate correlation of features with d1 and d2
# Helper to compute correlation
def get_corr(t1, t2):
    # centered
    t1_c = t1 - t1.mean()
    t2_c = t2 - t2.mean()
    return (t1_c * t2_c).sum() / (t1_c.norm() * t2_c.norm())

corrs_d1 = []
corrs_d2 = []
corrs_diff = [] # Correlate with d1 - d2

for i in range(sae.cfg.d_sae):
    acts = sae_acts_all[:, i]
    if acts.sum() == 0:
        corrs_d1.append(0)
        corrs_d2.append(0)
        corrs_diff.append(0)
        continue
        
    corrs_d1.append(get_corr(acts, d1_all.float()).item())
    corrs_d2.append(get_corr(acts, d2_all.float()).item())
    corrs_diff.append(get_corr(acts, (d1_all - d2_all).float()).item())

corrs_d1 = torch.tensor(corrs_d1)
corrs_d2 = torch.tensor(corrs_d2)

# %%
# Find "d1" features (High correlation with d1, low with d2)
d1_features = torch.argsort(corrs_d1.abs() - corrs_d2.abs(), descending=True)[:3]

print("Potential 'd1' (Vertical) Features:")
for idx in d1_features:
    analyze_feature(idx.item())

# %%
# Find "d2" features (High correlation with d2, low with d1)
d2_features = torch.argsort(corrs_d2.abs() - corrs_d1.abs(), descending=True)[:3]

print("Potential 'd2' (Horizontal) Features:")
for idx in d2_features:
    analyze_feature(idx.item())

# %%
# Find "Comparison" features (Triangle shapes)
# We look for features that don't correlate strongly linearly, but are highly sparse (~50%)
# and not dead.
sparsity = (sae_acts_all > 0).float().mean(dim=0)
# Filter for features active roughly 40-60% of the time
candidate_mask = (sparsity > 0.4) & (sparsity < 0.6)
indices = torch.nonzero(candidate_mask).squeeze()

if len(indices.shape) > 0 and indices.numel() > 0:
    print("Potential Comparison/Triangle Features:")
    # Just plot the first few candidates
    for idx in indices[:3]:
        analyze_feature(idx.item())
else:
    print("No obvious 50% sparsity features found. Check manual exploration.")

# %%