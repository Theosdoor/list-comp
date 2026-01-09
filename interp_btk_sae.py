#%% [markdown]
# # SAE Validation: Order by Scale Paper
# 
# This script validates the key predictions from the "Order by Scale" paper about
# graded latent activations in Sparse Autoencoders trained on SEP token activations.
#
# **Key Predictions to Validate:**
# 1. SAE decoder directions align with (E_di + P_di) vectors
# 2. SAE latent activations correlate with attention probabilities (α_s→d1, α_s→d2)
# 3. Same features active for (a,b) and (b,a) but with different magnitudes (graded, not binary)
# 4. Relative magnitude encodes sequence order

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy import stats

# Import project utilities
from model_utils import make_model, configure_runtime, load_model
from data import get_dataset

# Set Device
device =  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
torch.set_grad_enabled(False) # don't need gradients - analysis only

#%% [markdown]
# ## 1. Configuration & Model Loading

#%%
# --- Configuration (Must match training) ---
class SAEConfig:
    d_model = 64
    d_sae = 256  # ~4× expansion (100 D1 + 100 D2 + buffer)
    k = 4  # TopK sparsity
    
    # Model Config (from paper Table 3)
    n_layers = 2
    n_heads = 1
    list_len = 2
    n_digits = 100
    sep_token_index = 2  # Position of SEP in [d1, d2, SEP, o1, o2]

cfg = SAEConfig()

# --- BatchTopK SAE Definition ---
class BatchTopKSAE(nn.Module):
    """BatchTopK Sparse Autoencoder (Bussmann et al. 2024)"""
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))

    def encode(self, x):
        """Encode with BatchTopK sparsity."""
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

#%%
# --- Load Models ---
MODEL_PATH = "models/2layer_100dig_64d.pt"

# Setup Runtime (required by model_utils)
configure_runtime(
    list_len=cfg.list_len,
    seq_len=2 * cfg.list_len + 1,  # [d1, d2, SEP, o1, o2] = 5
    vocab=cfg.n_digits + 2,  # digits + MASK + SEP
    device=device
)

# Load base transformer model (with required kwargs)
try:
    model = load_model(
        MODEL_PATH,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        ln=False,
        use_bias=False,
        use_wv=False,
        use_wo=False
    )
    print(f"✓ Loaded base model from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    raise

# Load SAE
SAE_PATH = "sae.pt"
sae_checkpoint = torch.load(SAE_PATH, map_location=device, weights_only=False)

sae = BatchTopKSAE(cfg).to(device)
sae.load_state_dict(sae_checkpoint["state_dict"])

# Load the mean for centering (critical for SAE)
act_mean = sae_checkpoint["act_mean"].to(device)

print(f"✓ Loaded SAE from {SAE_PATH}")
print(f"  - Latent dim: {cfg.d_sae}")
print(f"  - TopK: {cfg.k}")

#%% [markdown]
# ## 2. Collect Activations and Attention Patterns
# 
# From the paper, the SEP token residual after layer 1 is:
# ```
# r_s^L1 = α_s→d1 (E_d1 + P_d1) + α_s→d2 (E_d2 + P_d2) + E_s + P_s
# ```
# 
# We need to collect:
# - SEP token activations (to run through SAE)
# - Attention patterns α_s→d1 and α_s→d2

#%%
# Generate ALL (d1, d2) pairs for complete analysis
val_ds, _ = get_dataset(
    list_len=cfg.list_len,
    n_digits=cfg.n_digits,
    train_split=1.0,  # Get all data
    no_dupes=False    # Include d1 == d2 cases
)
val_dl = DataLoader(val_ds, batch_size=2048, shuffle=False)

# Storage
all_d1 = []
all_d2 = []
all_sep_acts = []       # SEP token activations
all_sae_acts = []       # SAE latent activations
all_alpha_d1 = []       # Attention: SEP -> d1
all_alpha_d2 = []       # Attention: SEP -> d2

print("Collecting activations and attention patterns...")

layer_idx = 0  # Layer 1 (0-indexed) - where composition happens
hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
hook_name_attn = f"blocks.{layer_idx}.attn.hook_pattern"

with torch.no_grad():
    for inputs, targets in tqdm(val_dl):
        inputs = inputs.to(device)
        
        # Extract d1, d2 values
        d1 = inputs[:, 0]
        d2 = inputs[:, 1]
        
        # Run model and cache activations
        _, cache = model.run_with_cache(
            inputs, 
            stop_at_layer=layer_idx + 1,
            names_filter=[hook_name_resid, hook_name_attn]
        )
        
        # Get SEP token activations: [batch, d_model]
        sep_acts = cache[hook_name_resid][:, cfg.sep_token_index, :]
        
        # Get attention pattern: [batch, n_heads, seq, seq]
        # We want attention FROM SEP (query) TO d1 and d2 (keys)
        attn_pattern = cache[hook_name_attn][:, 0, :, :]  # [batch, seq, seq] (single head)
        alpha_d1 = attn_pattern[:, cfg.sep_token_index, 0]  # SEP attending to d1
        alpha_d2 = attn_pattern[:, cfg.sep_token_index, 1]  # SEP attending to d2
        
        # Run SAE encoding (with mean centering)
        sep_acts_centered = sep_acts - act_mean
        _, sae_z = sae(sep_acts_centered)
        
        # Store
        all_d1.append(d1.cpu())
        all_d2.append(d2.cpu())
        all_sep_acts.append(sep_acts.cpu())
        all_sae_acts.append(sae_z.cpu())
        all_alpha_d1.append(alpha_d1.cpu())
        all_alpha_d2.append(alpha_d2.cpu())

# Concatenate
d1_all = torch.cat(all_d1)
d2_all = torch.cat(all_d2)
sep_acts_all = torch.cat(all_sep_acts)
sae_acts_all = torch.cat(all_sae_acts)
alpha_d1_all = torch.cat(all_alpha_d1)
alpha_d2_all = torch.cat(all_alpha_d2)

n_samples = len(d1_all)
print(f"\n✓ Collected {n_samples} samples")
print(f"  - SEP activations shape: {sep_acts_all.shape}")
print(f"  - SAE activations shape: {sae_acts_all.shape}")

#%% [markdown]
# ## 3. Basic SAE Metrics

#%%
print("=" * 60)
print("SAE METRICS")
print("=" * 60)

# L0 Sparsity
l0 = (sae_acts_all > 0).float().sum(dim=1).mean()
print(f"Average L0 (active features per sample): {l0:.2f}")

# Dead features
dead_features = (sae_acts_all.sum(dim=0) == 0).sum().item()
print(f"Dead features: {dead_features} / {cfg.d_sae} ({100*dead_features/cfg.d_sae:.1f}%)")

# Feature firing rates
firing_rate = (sae_acts_all > 0).float().mean(dim=0)
print(f"Firing rate range: [{firing_rate[firing_rate > 0].min():.4f}, {firing_rate.max():.4f}]")

#%% [markdown]
# ## 4. Validation 1: Decoder-Embedding Alignment
# 
# **Prediction:** SAE learns decoder directions corresponding to D1 = E_d1 + P_d1 and D2 = E_d2 + P_d2
# 
# We compute cosine similarity between SAE decoder vectors and the (token + position) embeddings.

#%%
print("\n" + "=" * 60)
print("VALIDATION 1: Decoder-Embedding Alignment")
print("=" * 60)

# Get embedding matrices from base model
E = model.embed.W_E.detach()  # [vocab, d_model] - token embeddings
P = model.pos_embed.W_pos.detach()  # [seq_len, d_model] - position embeddings

# Compute D1 and D2 directions for all possible digit values
# D_i = E[d_i] + P[position_i]
# Position 0 = d1, Position 1 = d2
D1_directions = E[:cfg.n_digits] + P[0]  # [100, d_model]
D2_directions = E[:cfg.n_digits] + P[1]  # [100, d_model]

# Normalize decoder vectors for cosine similarity
W_dec_norm = F.normalize(sae.W_dec.detach(), dim=1)  # [d_sae, d_model]
D1_norm = F.normalize(D1_directions, dim=1)  # [100, d_model]
D2_norm = F.normalize(D2_directions, dim=1)  # [100, d_model]

# Compute max cosine similarity for each SAE feature
# For each decoder direction, find best matching D1 or D2 direction
cos_D1 = W_dec_norm @ D1_norm.T  # [d_sae, 100]
cos_D2 = W_dec_norm @ D2_norm.T  # [d_sae, 100]

max_cos_D1 = cos_D1.abs().max(dim=1).values  # [d_sae]
max_cos_D2 = cos_D2.abs().max(dim=1).values  # [d_sae]

# Get best match overall
best_match = torch.maximum(max_cos_D1, max_cos_D2)

# Analyze top features by firing rate
top_k_features = 50
top_indices = torch.argsort(firing_rate, descending=True)[:top_k_features]
top_alignments = best_match[top_indices]

print(f"\nAlignment of top {top_k_features} features (by firing rate) with D = E + P directions:")
print(f"  Mean cosine similarity: {top_alignments.mean():.4f}")
print(f"  Max cosine similarity:  {top_alignments.max():.4f}")
print(f"  Features with alignment > 0.8: {(top_alignments > 0.8).sum().item()}")
print(f"  Features with alignment > 0.5: {(top_alignments > 0.5).sum().item()}")

#%% [markdown]
# ## 5. Validation 2: Activation-Attention Correlation
# 
# **Prediction:** SAE latent activations should equal attention probabilities (α_s→d1, α_s→d2)
# 
# For features aligned with D1, their activation should correlate with α_s→d1, and similarly for D2.

#%%
print("\n" + "=" * 60)
print("VALIDATION 2: Activation-Attention Correlation")
print("=" * 60)

# For each top feature, determine if it's more aligned with D1 or D2 pattern
# Then compute correlation with corresponding attention weight

# Features more aligned with D1 pattern (via position embedding)
is_D1_aligned = max_cos_D1 > max_cos_D2

correlations_with_alpha = []
feature_info = []

for feat_idx in top_indices[:20]:  # Analyze top 20 features
    feat_acts = sae_acts_all[:, feat_idx].numpy()
    
    if feat_acts.sum() == 0:  # Skip dead features
        continue
    
    # Correlate with alpha_d1 and alpha_d2
    corr_d1, p_d1 = stats.pearsonr(feat_acts, alpha_d1_all.numpy())
    corr_d2, p_d2 = stats.pearsonr(feat_acts, alpha_d2_all.numpy())
    
    # Expected: aligned features correlate with their respective alpha
    aligned_with = "D1" if is_D1_aligned[feat_idx] else "D2"
    expected_corr = corr_d1 if aligned_with == "D1" else corr_d2
    
    correlations_with_alpha.append({
        'feature': feat_idx.item(),
        'aligned_with': aligned_with,
        'corr_alpha_d1': corr_d1,
        'corr_alpha_d2': corr_d2,
        'expected_corr': expected_corr,
        'alignment': best_match[feat_idx].item()
    })

corr_df = pd.DataFrame(correlations_with_alpha)
print("\nTop features correlation with attention patterns:")
print(corr_df.to_string(index=False))

avg_expected_corr = corr_df['expected_corr'].mean()
print(f"\n✓ Average correlation with expected alpha: {avg_expected_corr:.4f}")

#%% [markdown]
# ## 6. Validation 3: Graded Activations (Same Features, Different Magnitudes)
# 
# **Prediction:** The same features are active for (a,b) and (b,a), but with different magnitudes.
# This contradicts the binary view of SAE features.

#%%
print("\n" + "=" * 60)
print("VALIDATION 3: Graded Activations for Swapped Inputs")
print("=" * 60)

# For pairs (a, b) and (b, a), compare which features fire and their magnitudes
# We need to find matching pairs where d1=a, d2=b and d1=b, d2=a

# Create index for fast lookup
pair_to_idx = {}
for i in range(n_samples):
    pair_to_idx[(d1_all[i].item(), d2_all[i].item())] = i

# Find all swappable pairs (where a != b)
swapped_comparisons = []
for i in range(n_samples):
    a, b = d1_all[i].item(), d2_all[i].item()
    if a >= b:  # Only process each pair once, skip a==b
        continue
    
    if (b, a) in pair_to_idx:
        j = pair_to_idx[(b, a)]
        
        acts_ab = sae_acts_all[i]  # Features for (a, b)
        acts_ba = sae_acts_all[j]  # Features for (b, a)
        
        # Find features active in BOTH
        both_active = (acts_ab > 0) & (acts_ba > 0)
        n_both = both_active.sum().item()
        
        # Among features active in both, what's the magnitude difference?
        if n_both > 0:
            ratio = acts_ab[both_active] / acts_ba[both_active]
            
            swapped_comparisons.append({
                'a': a, 'b': b,
                'n_shared_features': n_both,
                'n_ab_only': ((acts_ab > 0) & (acts_ba == 0)).sum().item(),
                'n_ba_only': ((acts_ab == 0) & (acts_ba > 0)).sum().item(),
                'mean_ratio': ratio.mean().item(),
                'min_ratio': ratio.min().item(),
                'max_ratio': ratio.max().item(),
            })

swap_df = pd.DataFrame(swapped_comparisons)

print(f"\nAnalyzed {len(swap_df)} swapped pairs (a,b) vs (b,a)")
print(f"\nShared features (active in BOTH swapped inputs):")
print(f"  Mean count: {swap_df['n_shared_features'].mean():.1f}")
print(f"  Median count: {swap_df['n_shared_features'].median():.1f}")
print(f"\nMagnitude ratios (for shared features):")
print(f"  Mean ratio: {swap_df['mean_ratio'].mean():.4f}")
print(f"  Std of mean ratio: {swap_df['mean_ratio'].std():.4f}")
print(f"\nUnique features (active in only one ordering):")
print(f"  Mean (a,b) only: {swap_df['n_ab_only'].mean():.1f}")
print(f"  Mean (b,a) only: {swap_df['n_ba_only'].mean():.1f}")

# Key finding: if many features are shared but with different magnitudes, this supports graded activations
graded_evidence = swap_df['n_shared_features'].mean() > swap_df['n_ab_only'].mean()
print(f"\n✓ Graded activation evidence: {'SUPPORTED' if graded_evidence else 'NOT SUPPORTED'}")
print(f"  (More shared features than unique features indicates graded, not binary, representation)")

#%% [markdown]
# ## 7. Validation 4: Relative Magnitude Encodes Order
# 
# **Prediction:** The order of inputs (which is larger) is encoded by the relative magnitude
# of feature activations, not by which features fire.

#%%
print("\n" + "=" * 60)
print("VALIDATION 4: Relative Magnitude Encodes Order")
print("=" * 60)

# For the most frequent features, check if activation magnitude correlates with input ordering
# Specifically: does act_d1_feature - act_d2_feature correlate with d1 - d2?

# Find pairs of features: one aligned with D1 position, one with D2 position
# that often fire together

# Get feature pairs that co-occur frequently
is_D1_aligned_cpu = is_D1_aligned.cpu()
top_D1_features = top_indices[is_D1_aligned_cpu[top_indices]][:10]
top_D2_features = top_indices[~is_D1_aligned_cpu[top_indices]][:10]

print(f"\nAnalyzing feature pairs (D1-aligned vs D2-aligned):")

order_correlations = []
for d1_feat in top_D1_features:
    for d2_feat in top_D2_features:
        # Get activations for both features
        d1_acts = sae_acts_all[:, d1_feat].numpy()
        d2_acts = sae_acts_all[:, d2_feat].numpy()
        
        # Only consider samples where both are active
        both_active_mask = (d1_acts > 0) & (d2_acts > 0)
        if both_active_mask.sum() < 100:  # Need enough samples
            continue
        
        # Relative activation
        relative_act = d1_acts[both_active_mask] - d2_acts[both_active_mask]
        
        # Input ordering
        input_diff = (d1_all.numpy() - d2_all.numpy())[both_active_mask]
        
        # Correlation
        corr, p_val = stats.pearsonr(relative_act, input_diff)
        
        if abs(corr) > 0.3:  # Only record meaningful correlations
            order_correlations.append({
                'D1_feature': d1_feat.item(),
                'D2_feature': d2_feat.item(),
                'correlation': corr,
                'p_value': p_val,
                'n_samples': both_active_mask.sum()
            })

if order_correlations:
    order_df = pd.DataFrame(order_correlations)
    order_df = order_df.sort_values('correlation', key=abs, ascending=False)
    
    print(order_df.head(10).to_string(index=False))
    print(f"\n✓ Found {len(order_df)} feature pairs where relative activation encodes order")
    print(f"  Best correlation: {order_df['correlation'].abs().max():.4f}")
else:
    print("No strong order-encoding correlations found in top feature pairs")

#%% [markdown]
# ## 8. Visualization: Feature Activation Heatmaps

#%%
def plot_feature_heatmap(feature_idx, title_suffix=""):
    """Plot feature activation as heatmap over (d1, d2) grid."""
    acts = sae_acts_all[:, feature_idx].numpy()
    
    if acts.max() == 0:
        print(f"Feature {feature_idx} is dead.")
        return
    
    df = pd.DataFrame({
        'd1': d1_all.numpy(),
        'd2': d2_all.numpy(),
        'activation': acts
    })
    
    pivot = df.pivot_table(index='d1', columns='d2', values='activation', aggfunc='mean')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    ax.invert_yaxis()
    ax.set_title(f"Feature {feature_idx} Activation{title_suffix}\n(Y=d1, X=d2)")
    ax.set_xlabel("d2")
    ax.set_ylabel("d1")
    plt.tight_layout()
    plt.savefig(f"feature_{feature_idx}_heatmap.png", dpi=150)
    plt.show()
    
    # Analyze pattern
    # Vertical stripes = responds to d1, Horizontal stripes = responds to d2
    row_var = pivot.var(axis=1).mean()  # Variance across d2 for each d1
    col_var = pivot.var(axis=0).mean()  # Variance across d1 for each d2
    
    print(f"  Row variance (d1 selectivity): {row_var:.4f}")
    print(f"  Col variance (d2 selectivity): {col_var:.4f}")
    print(f"  → Feature responds more to: {'d1' if row_var < col_var else 'd2'}")

# Plot top 3 most frequent features
print("\n" + "=" * 60)
print("FEATURE HEATMAPS (Top 3 by Firing Rate)")
print("=" * 60)

for idx in top_indices[:3]:
    print(f"\nFeature {idx.item()}:")
    plot_feature_heatmap(idx.item())

#%% [markdown]
# ## 9. Visualization: Activation vs Attention Scatter

#%%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pick a highly aligned feature
best_feat_idx = top_indices[0].item()
feat_acts = sae_acts_all[:, best_feat_idx].numpy()

# Scatter: activation vs alpha_d1
axes[0].scatter(alpha_d1_all.numpy(), feat_acts, alpha=0.1, s=1)
axes[0].set_xlabel("α_s→d1 (Attention to d1)")
axes[0].set_ylabel(f"Feature {best_feat_idx} Activation")
axes[0].set_title(f"Feature Activation vs Attention to d1")

# Scatter: activation vs alpha_d2
axes[1].scatter(alpha_d2_all.numpy(), feat_acts, alpha=0.1, s=1)
axes[1].set_xlabel("α_s→d2 (Attention to d2)")
axes[1].set_ylabel(f"Feature {best_feat_idx} Activation")
axes[1].set_title(f"Feature Activation vs Attention to d2")

plt.tight_layout()
plt.savefig("activation_vs_attention.png", dpi=150)
plt.show()

#%% [markdown]
# ## 10. Summary

#%%
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print("""
The Order by Scale paper predicts that:

1. SAE decoder directions should align with (E_di + P_di) vectors
   → This determines whether a feature encodes d1-position or d2-position info

2. SAE latent activations should correlate with attention probabilities
   → Features encoding d1/d2 should activate proportionally to α_s→d1 / α_s→d2

3. Same features active for (a,b) and (b,a) with different magnitudes
   → CONTRADICTS the binary view of SAE features
   → Features are GRADED, not binary on/off

4. Relative magnitude encodes sequence order
   → Can't determine order from WHICH features fire
   → Must compare activation VALUES between d1-features and d2-features

If these validations pass, it supports the paper's claim that:
"The binary perspective of SAE latents is insufficient to explain the computation"
""")

print("See generated plots for visual confirmation.")
print("=" * 60)