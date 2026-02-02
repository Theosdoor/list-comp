"""
W&B Sweep Script for SAE Hyperparameter Search

Runs grid search over SAE configurations with multiple seeds.
Each configuration is trained 3 times with different random seeds.
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt

from dictionary_learning.trainers import BatchTopKTrainer

from model_scripts.model_utils import make_model, configure_runtime, parse_model_name_safe
from model_scripts.data import get_dataset
from model_scripts.sae_analysis import (
    collect_sae_activations, 
    create_feature_heatmaps,
    compute_reconstruction_metrics,
    collect_attention_patterns,
    identify_special_features,
    create_firing_rate_histogram,
)


# Base Model Configuration
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)
SAVE_FOLDER = 'results/sae_models/sweep_runs'

# Base Model Config (derived from model name)
N_LAYERS = MODEL_CFG.n_layers
N_HEADS = 1
LIST_LEN = 2
N_DIGITS = MODEL_CFG.n_digits
D_MODEL = MODEL_CFG.d_model
SEP_TOKEN_INDEX = 2

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def get_sep_activations(model, dataloader, layer_idx=0, sep_idx=2, max_acts=100_000):
    """Extract SEP token activations from specified layer."""
    activations = []
    count = 0
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    
    print(f"Collecting activations from {hook_name} at SEP token (idx {sep_idx})...")
    
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Collecting", leave=False):
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


def train_sae_sweep():
    """Train SAE with W&B sweep configuration."""
    
    # Initialize W&B run
    run = wandb.init()
    config = wandb.config
    
    # Extract hyperparameters
    d_sae = config.d_sae
    top_k = config.top_k
    lr = config.lr
    n_steps = config.n_steps
    warmup_steps = config.warmup_steps
    batch_size = config.batch_size
    seed = config.seed
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Training SAE: d_sae={d_sae}, k={top_k}, lr={lr}, seed={seed}")
    print(f"{'='*60}\n")
    
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
    
    model_path = f"models/{MODEL_NAME}.pt"
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
    
    # Center activations
    act_mean = all_acts.mean(0)
    all_acts_centered = all_acts - act_mean
    
    print(f"Collected {len(all_acts)} activations, shape: {all_acts.shape}")
    
    sae_dl = DataLoader(all_acts_centered, batch_size=batch_size, shuffle=True)
    
    # 4. Initialize Trainer
    trainer = BatchTopKTrainer(
        steps=n_steps,
        activation_dim=D_MODEL,
        dict_size=d_sae,
        k=top_k,
        layer=0,
        lm_name="custom",
        lr=lr,
        warmup_steps=warmup_steps,
        seed=seed,
        device=DEVICE,
    )
    
    print(f"\nSAE Config: d_sae={d_sae}, k={top_k}")
    print(f"Training for {n_steps} steps...")
    
    # 5. Training Loop
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    
    iter_dl = cycle(sae_dl)
    pbar = tqdm(range(n_steps))
    
    for step in pbar:
        batch_acts = next(iter_dl)
        loss = trainer.update(step, batch_acts)
        
        # Log to W&B
        if step % 100 == 0:
            log_info = trainer.get_logging_parameters()
            effective_l0 = log_info.get('effective_l0', 0)
            
            wandb.log({
                "loss": loss,
                "effective_l0": effective_l0,
                "step": step,
            })
            
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "L0": f"{effective_l0:.1f}",
            })
    
    # 6. Final metrics
    log_info = trainer.get_logging_parameters()
    final_l0 = log_info.get('effective_l0', 0)
    
    wandb.summary["final_loss"] = loss
    wandb.summary["final_l0"] = final_l0
    
    # 6b. Generate and log feature heatmaps
    print("\nGenerating analysis metrics...")
    val_ds, _ = get_dataset(
        list_len=LIST_LEN,
        n_digits=N_DIGITS,
        train_split=1.0,
        no_dupes=False
    )
    val_dl = DataLoader(val_ds, batch_size=2048, shuffle=False)
    
    sae = trainer.ae
    
    # Collect SAE activations
    d1_all, d2_all, sae_acts_all = collect_sae_activations(
        model, sae, val_dl, act_mean,
        layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE
    )
    
    # 1. Feature heatmaps
    print("  - Creating feature heatmaps...")
    fig = create_feature_heatmaps(d1_all, d2_all, sae_acts_all, n_digits=N_DIGITS)
    wandb.log({"feature_heatmaps": wandb.Image(fig)})
    plt.close(fig)
    
    # 2. Basic sparsity metrics
    l0 = (sae_acts_all > 0).float().sum(dim=1).mean()
    dead_features = (sae_acts_all.sum(dim=0) == 0).sum().item()
    firing_rate = (sae_acts_all > 0).float().mean(dim=0)
    
    wandb.summary["avg_l0"] = l0.item()
    wandb.summary["dead_features"] = dead_features
    wandb.summary["dead_features_pct"] = 100 * dead_features / d_sae
    
    # 3. Reconstruction quality
    print("  - Computing reconstruction metrics...")
    recon_metrics = compute_reconstruction_metrics(
        model, sae, val_dl, act_mean,
        layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE
    )
    wandb.summary["reconstruction_mse"] = recon_metrics["mse"]
    wandb.summary["explained_variance"] = recon_metrics["explained_variance"]
    
    # 4. Special features (correlated with attention)
    print("  - Identifying special features...")
    alpha_d1_all, alpha_d2_all = collect_attention_patterns(
        model, val_dl, layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE
    )
    special_features_info = identify_special_features(
        sae_acts_all, alpha_d1_all, alpha_d2_all, threshold=0.5
    )
    
    wandb.summary["n_special_features"] = special_features_info["n_special_features"]
    wandb.summary["special_features_pct"] = 100 * special_features_info["n_special_features"] / d_sae
    wandb.summary["max_attn_correlation"] = special_features_info["max_correlation"]
    wandb.summary["mean_abs_attn_correlation"] = special_features_info["mean_abs_correlation"]
    
    # Log special features as table
    if special_features_info["special_features"]:
        special_features_table = wandb.Table(
            columns=["feature_idx", "correlation", "type"],
            data=[[f["feature_idx"], f["correlation"], f["type"]] 
                  for f in special_features_info["special_features"]]
        )
        wandb.log({"special_features": special_features_table})
    
    # 5. Firing rate histogram
    print("  - Creating firing rate histogram...")
    fig_firing = create_firing_rate_histogram(sae_acts_all)
    wandb.log({"firing_rate_histogram": wandb.Image(fig_firing)})
    plt.close(fig_firing)
    
    print("✓ Logged all analysis metrics to W&B")
    
    # 7. Save SAE
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    save_name = f'sae_d{d_sae}_k{top_k}_lr{lr}_seed{seed}_{MODEL_NAME}.pt'
    save_path = os.path.join(SAVE_FOLDER, save_name)
    
    sae = trainer.ae
    checkpoint = {
        "state_dict": sae.state_dict(),
        "cfg": {
            "activation_dim": D_MODEL,
            "dict_size": d_sae,
            "k": top_k,
            "d_model": D_MODEL,
            "d_sae": d_sae,
            "lr": lr,
            "seed": seed,
        },
        "act_mean": act_mean.cpu(),
        "final_loss": loss,
        "final_l0": final_l0,
    }
    
    torch.save(checkpoint, save_path)
    print(f"\n✓ SAE saved to {save_path}")
    
    # Save model as W&B artifact
    artifact = wandb.Artifact(
        name=f"sae-d{d_sae}-k{top_k}-seed{seed}",
        type="model",
        metadata={
            "d_sae": d_sae,
            "top_k": top_k,
            "lr": lr,
            "seed": seed,
            "final_loss": loss,
            "final_l0": final_l0,
        }
    )
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()


if __name__ == "__main__":
    train_sae_sweep()
