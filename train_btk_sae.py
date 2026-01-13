#%% [markdown]
# # Train BatchTopK SAE on SEP Token Activations
# 
# Uses BatchTopKTrainer from saprmarks/dictionary_learning library.
# Trains an SAE with config optimized for the Order by Scale paper predictions.

#%%
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dictionary_learning.trainers import BatchTopKTrainer

from model_utils import make_model, configure_runtime, parse_model_name_safe
from data import get_dataset

#%%
# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)
SAVE_PATH = 'sae_models/sae2.pt'

class SAEConfig:
    # Architecture
    d_model = MODEL_CFG.d_model      # activation_dim
    d_sae = 256                      # dict_size (~4× expansion)
    k = 4                            # top_k: activations per sample
    
    # Training
    lr = 3e-4
    batch_size = 4096
    n_steps = 10_000
    warmup_steps = 1000
    
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
    
    # 4. Initialize Trainer (handles SAE creation internally)
    trainer = BatchTopKTrainer(
        steps=cfg.n_steps,
        activation_dim=cfg.d_model,
        dict_size=cfg.d_sae,
        k=cfg.k,
        layer=0,  # Required but we're using custom activations
        lm_name="custom",  # Required but we're using custom activations
        lr=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        seed=cfg.seed,
        device=cfg.device,
    )
    
    print(f"\nSAE Config: d_sae={cfg.d_sae}, k={cfg.k}")
    print(f"Training for {cfg.n_steps} steps...")
    
    # 5. Training Loop using trainer.update()
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    
    iter_dl = cycle(sae_dl)
    pbar = tqdm(range(cfg.n_steps))
    
    for step in pbar:
        batch_acts = next(iter_dl)
        
        # Trainer's update() returns the loss value
        loss = trainer.update(step, batch_acts)
        
        if step % 100 == 0:
            # Get logging info - library uses 'effective_l0', not 'l0'
            log_info = trainer.get_logging_parameters()
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "L0": f"{log_info.get('effective_l0', 0):.1f}",
            })
    
    # 6. Save - get SAE from trainer
    sae = trainer.ae  # The trained autoencoder
    
    checkpoint = {
        "state_dict": sae.state_dict(),
        "cfg": {
            "activation_dim": cfg.d_model,
            "dict_size": cfg.d_sae,
            "k": cfg.k,
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