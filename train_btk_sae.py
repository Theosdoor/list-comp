#%% [markdown]
# # Train BatchTopK SAE on SEP Token Activations
# 
# Uses BatchTopKTrainer from saprmarks/dictionary_learning library.
# Trains an SAE with config optimized for the Order by Scale paper predictions.

#%%
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dictionary_learning.trainers import BatchTopKTrainer
# https://github.com/saprmarks/dictionary_learning/blob/main/dictionary_learning/trainers/batch_top_k.py

from model_utils import make_model, configure_runtime, parse_model_name_safe
from data import get_dataset

#%%
# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)
SAVE_FOLDER = 'sae_models'


# Architecture
D_MODEL = MODEL_CFG.d_model      # activation_dim
D_SAE = 150                      # dict_size
TOP_K = 4                            # top_k: activations per sample

# Training
LR = 3e-4
BATCH_SIZE = 4096
N_STEPS = 10_000
WARMUP_STEPS = 1000

# Base Model Config (derived from model name)
N_LAYERS = MODEL_CFG.n_layers
N_HEADS = 1
LIST_LEN = 2
N_DIGITS = MODEL_CFG.n_digits
SEP_TOKEN_INDEX = 2               # [d1, d2, SEP, o1, o2] -> Index 2

# Runtime
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

SAVE_NAME = f'sae_d{D_SAE}_k{TOP_K}_{N_STEPS//1000}ksteps_{MODEL_NAME}.pt'
SAVE_PATH = os.path.join(SAVE_FOLDER, SAVE_NAME)


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
    print(f"Using device: {DEVICE}")
    
    # 1. Load Base Model
    print("Loading base model...")
    configure_runtime(
        list_len=LIST_LEN, 
        seq_len=LIST_LEN * 2 + 1, 
        vocab=N_DIGITS + 2, 
        device=DEVICE
    )
    
    model = make_model(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        ln=False,
        use_bias=False,
        use_wv=False,
        use_wo=False
    )
    
    model_path = "models/" + MODEL_NAME + ".pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✓ Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # 2. Prepare Data
    train_ds, _ = get_dataset(
        list_len=LIST_LEN,
        n_digits=N_DIGITS,
        mask_tok=N_DIGITS,
        sep_tok=N_DIGITS + 1
    )
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
    
    # 3. Collect Activations
    all_acts = get_sep_activations(
        model, train_dl, 
        layer_idx=0, 
        sep_idx=SEP_TOKEN_INDEX
    )
    all_acts = all_acts.to(DEVICE)
    
    # Center activations (important for SAE)
    act_mean = all_acts.mean(0)
    all_acts_centered = all_acts - act_mean
    
    print(f"Collected {len(all_acts)} activations, shape: {all_acts.shape}")
    
    sae_dl = DataLoader(all_acts_centered, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Initialize Trainer (handles SAE creation internally)
    trainer = BatchTopKTrainer(
        steps=N_STEPS,
        activation_dim=D_MODEL,
        dict_size=D_SAE,
        k=TOP_K,
        layer=0,  # Required but we're using custom activations
        lm_name="custom",  # Required but we're using custom activations
        lr=LR,
        warmup_steps=WARMUP_STEPS,
        seed=SEED,
        device=DEVICE,
    )
    
    print(f"\nSAE Config: d_sae={D_SAE}, k={TOP_K}")
    print(f"Training for {N_STEPS} steps...")
    
    # 5. Training Loop using trainer.update()
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    
    iter_dl = cycle(sae_dl)
    pbar = tqdm(range(N_STEPS))
    
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
            "activation_dim": D_MODEL,
            "dict_size": D_SAE,
            "k": TOP_K,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
        },
        "act_mean": act_mean.cpu()
    }
    
    torch.save(checkpoint, SAVE_PATH)
    print(f"\n✓ SAE saved to {SAVE_PATH}")
    print(f"  Config: d_sae={D_SAE}, k={TOP_K}")

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BatchTopK SAE")
    parser.add_argument("--d_sae", type=int, default=D_SAE, help=f"Dictionary size (default: {D_SAE})")
    parser.add_argument("--top_k", type=int, default=TOP_K, help=f"TopK sparsity (default: {TOP_K})")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default: {LR})")
    parser.add_argument("--n_steps", type=int, default=N_STEPS, help=f"Training steps (default: {N_STEPS})")
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS, help=f"Warmup steps (default: {WARMUP_STEPS})")
    args = parser.parse_args()
    
    # Update globals with command-line overrides
    D_SAE = args.d_sae
    TOP_K = args.top_k
    LR = args.lr
    N_STEPS = args.n_steps
    WARMUP_STEPS = args.warmup_steps
    
    # Update save path with new config
    SAVE_NAME = f'sae_d{D_SAE}_k{TOP_K}_{MODEL_NAME}.pt'
    SAVE_PATH = os.path.join(SAVE_FOLDER, SAVE_NAME)
    
    train_sae()