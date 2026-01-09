#%% [markdown]
# # Train BatchTopK SAE on SEP Token Activations
# 
# Trains an SAE with config optimized for the Order by Scale paper predictions:
# - d_sae = 256 (~100 D1 + 100 D2 + buffer)
# - k = 4 (sparse: only relevant digit features should fire)

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model_utils import make_model, configure_runtime
from data import get_dataset

#%%
# --- Configuration ---
class SAEConfig:
    d_model = 64
    d_sae = 256         # ~4× expansion (100 D1 + 100 D2 + buffer)
    k = 4               # Sparse: only 2-4 features should matter per input
    
    # Training
    lr = 3e-4
    batch_size = 4096
    n_steps = 10_000
    
    # Base Model Config (must match trained model)
    n_layers = 2
    n_heads = 1
    list_len = 2
    n_digits = 100
    sep_token_index = 2  # [d1, d2, SEP, o1, o2] -> Index 2

#%%
# --- BatchTopK SAE Implementation ---
class BatchTopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(cfg.d_model, cfg.d_sae)
        ))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(cfg.d_sae, cfg.d_model)
        ))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        
        self.set_decoder_norm_to_unit_norm()

    def set_decoder_norm_to_unit_norm(self):
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8)

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

#%%
def get_sep_activations(model, dataloader, layer_idx=0, max_acts=100_000):
    """Extract SEP token activations from layer 1 (where composition happens)."""
    activations = []
    count = 0
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    
    print(f"Collecting activations from {hook_name} at SEP token (idx {SAEConfig.sep_token_index})...")
    
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Collecting"):
            inputs = inputs.to(model.cfg.device)
            _, cache = model.run_with_cache(inputs, stop_at_layer=layer_idx+1, names_filter=hook_name)
            sep_acts = cache[hook_name][:, SAEConfig.sep_token_index, :]
            activations.append(sep_acts.cpu())
            count += sep_acts.shape[0]
            if count >= max_acts:
                break
    
    return torch.cat(activations, dim=0)

#%%
def train_sae():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cfg = SAEConfig()
    
    # 1. Load Base Model
    print("Loading base model...")
    configure_runtime(
        list_len=cfg.list_len, 
        seq_len=cfg.list_len * 2 + 1, 
        vocab=cfg.n_digits + 2, 
        device=device
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
    model_path = "models/2layer_100dig_64d.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
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
    all_acts = get_sep_activations(model, train_dl, layer_idx=0)
    all_acts = all_acts.to(device)
    
    # Center activations
    act_mean = all_acts.mean(0)
    all_acts = all_acts - act_mean
    
    print(f"Collected {len(all_acts)} activations, shape: {all_acts.shape}")
    
    sae_dl = DataLoader(all_acts, batch_size=cfg.batch_size, shuffle=True)
    
    # 4. Initialize SAE
    sae = BatchTopKSAE(cfg).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    
    print(f"\nSAE Config: d_sae={cfg.d_sae}, k={cfg.k}")
    print(f"Training for {cfg.n_steps} steps...")
    
    # 5. Train
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    
    iter_dl = cycle(sae_dl)
    pbar = tqdm(range(cfg.n_steps))
    
    for step in pbar:
        batch_acts = next(iter_dl)
        
        x_reconstruct, z = sae(batch_acts)
        loss = F.mse_loss(x_reconstruct, batch_acts)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        optimizer.step()
        
        sae.set_decoder_norm_to_unit_norm()
        
        if step % 100 == 0:
            l0 = (z > 0).float().sum(dim=1).mean().item()
            pbar.set_postfix({"mse": f"{loss.item():.6f}", "L0": f"{l0:.1f}"})
    
    # 6. Save
    checkpoint = {
        "state_dict": sae.state_dict(),
        "cfg": vars(cfg),
        "act_mean": act_mean.cpu()
    }
    
    save_path = "sae.pt"
    torch.save(checkpoint, save_path)
    print(f"\n✓ SAE saved to {save_path}")
    print(f"  Config: d_sae={cfg.d_sae}, k={cfg.k}")

#%%
if __name__ == "__main__":
    train_sae()