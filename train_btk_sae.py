#%% [markdown]
# # Train BatchTopK SAE on SEP Token Activations
# 
# Implementation based on: https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
# Trains an SAE with config optimized for the Order by Scale paper predictions.

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model_utils import make_model, configure_runtime, parse_model_name_safe
from data import get_dataset

#%%
# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)

class SAEConfig:
    # Architecture
    d_model = MODEL_CFG.d_model      # act_size in reference
    d_sae = 256                      # dict_size in reference (~4× expansion)
    k = 4                            # top_k: activations per sample
    
    # Training
    lr = 3e-4
    beta1 = 0.9
    beta2 = 0.999
    batch_size = 4096
    n_steps = 10_000
    max_grad_norm = 1.0
    
    # Loss coefficients
    l1_coeff = 0.0                    # L1 penalty (TopK handles sparsity, so can be 0)
    aux_penalty = 1/32                # Auxiliary loss coefficient for dead features
    
    # Dead feature detection
    n_batches_to_dead = 10            # Feature is "dead" if inactive for this many batches
    k_aux = 32                        # TopK for auxiliary loss
    
    # Base Model Config (derived from model name)
    n_layers = MODEL_CFG.n_layers
    n_heads = 1
    list_len = 2
    n_digits = MODEL_CFG.n_digits
    sep_token_index = 2               # [d1, d2, SEP, o1, o2] -> Index 2
    
    # Runtime
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#%%
# --- BatchTopK SAE Implementation (matching reference) ---

class BatchTopKSAE(nn.Module):
    """
    BatchTopK Sparse Autoencoder (Bussmann et al. 2024)
    
    Key difference from TopKSAE: selects top k*batch_size activations
    across the entire flattened batch, not per-sample.
    """
    
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        
        # Encoder/decoder weights and biases
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg.d_model, cfg.d_sae)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg.d_sae, cfg.d_model)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        
        # Initialize W_dec as normalized transpose of W_enc (per reference)
        with torch.no_grad():
            self.W_dec.data[:] = self.W_enc.t().data
            self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        
        # Dead feature tracking
        self.register_buffer(
            'num_batches_not_active', 
            torch.zeros(cfg.d_sae)
        )
    
    def encode(self, x):
        """Encode with batch-level TopK sparsity."""
        x_cent = x - self.b_dec
        pre_acts = x_cent @ self.W_enc + self.b_enc # TODO include b_enc? ref code doesnt but paper seems to
        acts = F.relu(pre_acts)
        
        # BatchTopK: select top k*batch_size across entire flattened batch
        batch_size = x.shape[0]
        total_k = self.cfg.k * batch_size
        
        acts_flat = acts.flatten()
        topk = torch.topk(acts_flat, total_k, dim=-1)
        
        acts_topk = torch.zeros_like(acts_flat)
        acts_topk.scatter_(-1, topk.indices, topk.values)
        acts_topk = acts_topk.reshape(acts.shape)
        
        return acts_topk, acts  # Return both sparse and full activations
    
    def decode(self, z):
        return z @ self.W_dec + self.b_dec
    
    def forward(self, x):
        """Forward pass returning loss dict (matching reference API)."""
        # TODO add preprocess_input from ref code? probs not needed
        acts_topk, acts = self.encode(x)
        x_reconstruct = self.decode(acts_topk)
        
        # Update dead feature tracking
        self.update_inactive_features(acts_topk)
        
        return self.get_loss_dict(x, x_reconstruct, acts, acts_topk)
    
    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk):
        """Compute all loss terms and metrics."""
        # Reconstruction loss (L2)
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        
        # Sparsity loss (L1 on activations)
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        
        # L0 (average number of active features per sample)
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        
        # Auxiliary loss for dead features
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        
        # Total loss
        loss = l2_loss + l1_loss + aux_loss
        
        # Count dead features
        num_dead_features = (self.num_batches_not_active > self.cfg.n_batches_to_dead).sum()
        # TODO include postprocess output? probs not needed
        return {
            "sae_out": x_reconstruct,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
    
    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        """
        Auxiliary loss to revive dead features.
        Uses dead features to reconstruct the residual.
        """
        dead_features = self.num_batches_not_active >= self.cfg.n_batches_to_dead
        
        if dead_features.sum() == 0:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        
        residual = x.float() - x_reconstruct.float()
        
        # TopK on dead feature activations only
        k_aux = min(self.cfg.k_aux, dead_features.sum().item())
        acts_dead = acts[:, dead_features]
        topk_aux = torch.topk(acts_dead, k_aux, dim=-1)
        
        acts_aux = torch.zeros_like(acts_dead)
        acts_aux.scatter_(-1, topk_aux.indices, topk_aux.values)
        
        # Reconstruct residual using dead features
        x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
        
        l2_loss_aux = self.cfg.aux_penalty * (
            (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        )
        
        return l2_loss_aux
    
    def update_inactive_features(self, acts):
        """Track which features haven't fired recently."""
        with torch.no_grad():
            # Increment counter for features that didn't fire
            self.num_batches_not_active += (acts.sum(0) == 0).float()
            # Reset counter for features that did fire
            self.num_batches_not_active[acts.sum(0) > 0] = 0
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """
        Project out radial component of decoder gradients and normalize weights.
        This keeps decoder vectors on the unit sphere during optimization.
        """
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        
        if self.W_dec.grad is not None:
            # Project out the component of gradient parallel to decoder direction
            W_dec_grad_proj = (
                (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            )
            self.W_dec.grad -= W_dec_grad_proj
        
        self.W_dec.data = W_dec_normed

#%%
def get_sep_activations(model, dataloader, layer_idx=0, sep_idx=2, max_acts=100_000):
    """Extract SEP token activations from specified layer."""
    activations = []
    count = 0
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    
    print(f"Collecting activations from {hook_name} at SEP token (idx {sep_idx})...")
    
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Collecting"):
            inputs = inputs.to(model.cfg.device)
            _, cache = model.run_with_cache(
                inputs, 
                stop_at_layer=layer_idx + 1, 
                names_filter=hook_name
            )
            sep_acts = cache[hook_name][:, sep_idx, :]
            activations.append(sep_acts.cpu())
            count += sep_acts.shape[0]
            if count >= max_acts:
                break
    
    return torch.cat(activations, dim=0)

#%%
def train_sae():
    cfg = SAEConfig()
    print(f"Using device: {cfg.device}")
    
    # 1. Load Base Model
    print("Loading base model...")
    configure_runtime(
        list_len=cfg.list_len, 
        seq_len=cfg.list_len * 2 + 1, 
        vocab=cfg.n_digits + 2, 
        device=cfg.device
    )
    
    model = make_model(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        ln=False,
        use_bias=False,
        use_wv=False,
        use_wo=False
    )
    
    import os
    model_path = "models/" + MODEL_NAME + ".pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        print(f"✓ Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # 2. Prepare Data
    train_ds, _ = get_dataset(
        list_len=cfg.list_len,
        n_digits=cfg.n_digits,
        mask_tok=cfg.n_digits,
        sep_tok=cfg.n_digits + 1
    )
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
    
    # 3. Collect Activations
    all_acts = get_sep_activations(
        model, train_dl, 
        layer_idx=0, 
        sep_idx=cfg.sep_token_index
    )
    all_acts = all_acts.to(cfg.device)
    
    # Center activations (important for SAE)
    act_mean = all_acts.mean(0)
    all_acts_centered = all_acts - act_mean
    
    print(f"Collected {len(all_acts)} activations, shape: {all_acts.shape}")
    
    sae_dl = DataLoader(all_acts_centered, batch_size=cfg.batch_size, shuffle=True)
    
    # 4. Initialize SAE
    sae = BatchTopKSAE(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(
        sae.parameters(), 
        lr=cfg.lr, 
        betas=(cfg.beta1, cfg.beta2)
    )
    
    print(f"\nSAE Config: d_sae={cfg.d_sae}, k={cfg.k}")
    print(f"Training for {cfg.n_steps} steps...")
    
    # 5. Training Loop
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    
    iter_dl = cycle(sae_dl)
    pbar = tqdm(range(cfg.n_steps))
    
    for step in pbar:
        batch_acts = next(iter_dl)
        
        # Forward pass
        output = sae(batch_acts)
        loss = output["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=cfg.max_grad_norm)
        
        # Project decoder gradients and normalize (per reference)
        sae.make_decoder_weights_and_grad_unit_norm()
        
        # Update weights and clear gradients
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 100 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "L2": f"{output['l2_loss'].item():.4f}",
                "L0": f"{output['l0_norm']:.1f}",
                "dead": f"{output['num_dead_features'].item():.0f}",
            })
    
    # 6. Save
    checkpoint = {
        "state_dict": sae.state_dict(),
        "cfg": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
        "act_mean": act_mean.cpu()
    }
    
    save_path = "sae.pt"
    torch.save(checkpoint, save_path)
    print(f"\n✓ SAE saved to {save_path}")
    print(f"  Config: d_sae={cfg.d_sae}, k={cfg.k}")
    print(f"  Final dead features: {output['num_dead_features'].item():.0f}/{cfg.d_sae}")

#%%
if __name__ == "__main__":
    train_sae()