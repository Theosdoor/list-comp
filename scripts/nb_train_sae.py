#%% [markdown]
# # Train BatchTopK SAE on SEP Token Activations
# 
# Uses BatchTopKTrainer from saprmarks/dictionary_learning library.
# Trains an SAE with config optimized for the Order by Scale paper predictions.

#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dictionary_learning.trainers import BatchTopKTrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer

from src.models.transformer import make_model, parse_model_name_safe
from src.utils.runtime import configure_runtime
from src.data.datasets import get_dataset

#%%
# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)
SAVE_FOLDER = 'results/sae_models'

# Architecture
D_MODEL = MODEL_CFG.d_model      # activation_dim
D_SAE = 150                      # dict_size
TOP_K = 4                        # top_k for btk/matryoshka
TARGET_L0 = 4.0                  # target sparsity for jumprelu
N_GROUPS = 4                     # number of nested groups for matryoshka
SAE_TYPE = 'btk'                 # sae type: btk | jumprelu | matryoshka

# Training
LR = 3e-4
BATCH_SIZE = 4096
N_STEPS = 50_000
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
    
    # 4. Initialize Trainer
    common = dict(steps=N_STEPS, activation_dim=D_MODEL, dict_size=D_SAE,
                  layer=0, lm_name="custom", lr=LR, warmup_steps=WARMUP_STEPS,
                  seed=SEED, device=DEVICE)

    if SAE_TYPE == "btk":
        trainer = BatchTopKTrainer(k=TOP_K, **common)
        extra_cfg = {"sae_type": "btk", "k": TOP_K}
        type_tag = f"k{TOP_K}"
    elif SAE_TYPE == "jumprelu":
        trainer = JumpReluTrainer(target_l0=TARGET_L0, **common)
        extra_cfg = {"sae_type": "jumprelu", "target_l0": TARGET_L0}
        type_tag = f"tl0{TARGET_L0}"
    elif SAE_TYPE == "matryoshka":
        frac = 1.0 / N_GROUPS
        group_fractions = [frac] * (N_GROUPS - 1) + [1.0 - frac * (N_GROUPS - 1)]
        trainer = MatryoshkaBatchTopKTrainer(k=TOP_K, group_fractions=group_fractions, **common)
        extra_cfg = {"sae_type": "matryoshka", "k": TOP_K, "n_groups": N_GROUPS,
                     "group_fractions": group_fractions, "group_sizes": trainer.group_sizes}
        type_tag = f"k{TOP_K}_ng{N_GROUPS}"
    else:
        raise ValueError(f"Unknown sae_type '{SAE_TYPE}'. Use btk, jumprelu, or matryoshka.")

    print(f"\nSAE type: {SAE_TYPE}  d_sae={D_SAE}  {type_tag}")
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
            log_info = trainer.get_logging_parameters()
            # effective_l0 is only exposed by BTK/Matryoshka trainers; for
            # JumpReLU compute it directly from the current batch
            if 'effective_l0' in log_info:
                l0_val = log_info['effective_l0']
            else:
                with torch.no_grad():
                    f = trainer.ae.encode(batch_acts.to(DEVICE))
                    l0_val = (f > 0).float().sum(dim=-1).mean().item()
            pbar.set_postfix({"loss": f"{loss:.4f}", "L0": f"{l0_val:.1f}"})
    
    # 6. Save - get SAE from trainer
    sae = trainer.ae
    save_name = f'{SAE_TYPE}_sae_d{D_SAE}_{type_tag}_{MODEL_NAME}.pt'
    save_path = os.path.join(SAVE_FOLDER, save_name)

    checkpoint = {
        "state_dict": sae.state_dict(),
        "cfg": {
            "activation_dim": D_MODEL,
            "dict_size": D_SAE,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            **extra_cfg,
        },
        "act_mean": act_mean.cpu()
    }

    torch.save(checkpoint, save_path)
    print(f"\n✓ SAE saved to {save_path}")

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE on SEP token activations")
    parser.add_argument("--sae_type", type=str, default=SAE_TYPE, choices=["btk", "jumprelu", "matryoshka"])
    parser.add_argument("--d_sae", type=int, default=D_SAE)
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k for btk/matryoshka")
    parser.add_argument("--target_l0", type=float, default=TARGET_L0, help="Target L0 for jumprelu")
    parser.add_argument("--n_groups", type=int, default=N_GROUPS, help="Number of nested groups for matryoshka")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    args = parser.parse_args()

    SAE_TYPE = args.sae_type
    D_SAE = args.d_sae
    TOP_K = args.top_k
    TARGET_L0 = args.target_l0
    N_GROUPS = args.n_groups
    LR = args.lr
    N_STEPS = args.n_steps
    WARMUP_STEPS = args.warmup_steps

    train_sae()