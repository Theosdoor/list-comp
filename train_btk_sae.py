#%% [markdown]
# # Train BatchTopK SAE on SEP Token Activations
# 
# Uses BatchTopKSAE from saprmarks/dictionary_learning library.
# Trains an SAE with config optimized for the Order by Scale paper predictions.

#%%
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

from model_utils import make_model, configure_runtime, parse_model_name_safe
from data import get_dataset

#%%
# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)
SAVE_PATH = 'sae2.pt'

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
    
    # 4. Initialize SAE using library
    # Library's BatchTopKSAE: (activation_dim, dict_size, k)
    sae = BatchTopKSAE(
        activation_dim=cfg.d_model,
        dict_size=cfg.d_sae,
        k=cfg.k
    ).to(cfg.device)
    
    optimizer = torch.optim.Adam(
        sae.parameters(), 
        lr=cfg.lr, 
        betas=(cfg.beta1, cfg.beta2)
    )
    
    print(f"\nSAE Config: d_sae={cfg.d_sae}, k={cfg.k}")
    print(f"Training for {cfg.n_steps} steps...")
    
    # 5. Training Loop (manual, since we have custom activation collection)
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    
    iter_dl = cycle(sae_dl)
    pbar = tqdm(range(cfg.n_steps))
    
    for step in pbar:
        batch_acts = next(iter_dl)
        
        # Library's forward: returns (reconstruction, features) with output_features=True
        x_reconstruct, features = sae(batch_acts, output_features=True)
        
        # Compute L2 reconstruction loss
        loss = (x_reconstruct - batch_acts).pow(2).mean()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=cfg.max_grad_norm)
        
        # Update weights and clear gradients
        optimizer.step()
        optimizer.zero_grad()
        
        # Compute L0 for logging
        l0 = (features > 0).float().sum(-1).mean()
        
        if step % 100 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "L0": f"{l0.item():.1f}",
            })
    
    # 6. Save - include library class info for loading
    checkpoint = {
        "state_dict": sae.state_dict(),
        "cfg": {
            "activation_dim": cfg.d_model,
            "dict_size": cfg.d_sae,
            "k": cfg.k,
            # Also save original cfg for compatibility
            "d_model": cfg.d_model,
            "d_sae": cfg.d_sae,
        },
        "act_mean": act_mean.cpu()
    }
    
    torch.save(checkpoint, SAVE_PATH)
    print(f"\n✓ SAE saved to {SAVE_PATH}")
    print(f"  Config: d_sae={cfg.d_sae}, k={cfg.k}")

#%%
if __name__ == "__main__":
    train_sae()