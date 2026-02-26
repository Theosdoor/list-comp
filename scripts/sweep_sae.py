"""
W&B Sweep Script for SAE Hyperparameter Search

Runs grid search over SAE configurations with multiple seeds.
Each configuration is trained 3 times with different random seeds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt

from dictionary_learning.trainers import BatchTopKTrainer

from src.models.transformer import make_model
from src.models.utils import infer_model_config
from src.utils.runtime import configure_runtime
from src.data.datasets import get_dataset
from src.sae.sae_analysis import (
    collect_sae_activations, 
    create_feature_heatmaps,
    compute_reconstruction_metrics,
    compute_sae_patched_accuracy,
    collect_attention_patterns,
    identify_special_features,
    create_firing_rate_histogram,
)
from train_sae import get_sep_activations


# Base Model Configuration
MODEL_NAME = '2layer_100dig_64d'
MODEL_PATH = f'models/{MODEL_NAME}.pt'
MODEL_CFG = infer_model_config(MODEL_PATH)
SAVE_FOLDER = 'results/sae_models/sweep_runs'

# Base Model Config (derived from checkpoint)
N_LAYERS = MODEL_CFG['n_layers']
N_HEADS = MODEL_CFG['n_heads']
D_MODEL = MODEL_CFG['d_model']
VOCAB = MODEL_CFG['d_vocab']
N_DIGITS = VOCAB - 2  # vocab = n_digits + mask_tok + sep_tok
LIST_LEN = 2  # Standard for this project
SEP_TOKEN_INDEX = 2

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


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
    
    # Generate meaningful run name matching the SAE checkpoint name
    sae_run_name = f'sae_d{d_sae}_k{top_k}_lr{lr}_seed{seed}'
    run.name = sae_run_name  # Set name on the run object returned by init()
    
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
    act_mean = all_acts.mean(0)  # Already on DEVICE since all_acts is on DEVICE
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
    try:
        # Use ALL data for analysis (not just validation split)
        all_data_ds, _ = get_dataset(
            list_len=LIST_LEN,
            n_digits=N_DIGITS,
            train_split=1.0,  # Use all data for comprehensive analysis
            no_dupes=False
        )
        analysis_dl = DataLoader(all_data_ds, batch_size=2048, shuffle=False)
        
        sae = trainer.ae
        sae = sae.to(DEVICE)  # Ensure SAE is on correct device
        
        # Collect SAE activations
        d1_all, d2_all, sae_acts_all = collect_sae_activations(
            model, sae, analysis_dl, act_mean,
            layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE
        )
        
        # 1. Feature heatmaps (skip for now)
        # print("  - Creating feature heatmaps...")
        # if d_sae <= 256:  # Only create if reasonable size
        #     fig = create_feature_heatmaps(d1_all, d2_all, sae_acts_all, n_digits=N_DIGITS)
        #     # Convert plotly figure to static image for W&B
        #     wandb.log({"feature_heatmaps": wandb.Plotly(fig)})
        #     print(f"    ✓ Logged feature heatmaps ({d_sae} features)")
        # else:
        #     print(f"    ⊘ Skipped heatmaps (d_sae={d_sae} too large for visualization)")
        
        # 2. Basic sparsity metrics
        l0 = (sae_acts_all > 0).float().sum(dim=1).mean()
        dead_features = (sae_acts_all.sum(dim=0) == 0).sum().item()
        firing_rate = (sae_acts_all > 0).float().mean(dim=0)
        
        wandb.summary["avg_l0"] = l0.item()
        wandb.summary["dead_features"] = dead_features
        wandb.summary["dead_features_pct"] = 100 * dead_features / d_sae
        
        # 3. Reconstruction quality
        print("  - Computing reconstruction metrics...")
        try:
            recon_metrics = compute_reconstruction_metrics(
                model, sae, analysis_dl, act_mean,
                layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE
            )
            wandb.summary["reconstruction_mse"] = recon_metrics["mse"]
            wandb.summary["explained_variance"] = recon_metrics["explained_variance"]
        except Exception as e:
            print(f"    ⚠ Warning: Could not compute reconstruction metrics - {e}")
            wandb.summary["reconstruction_mse"] = "NA"
            wandb.summary["explained_variance"] = "NA"
        
        # 3b. SAE reconstruction accuracy
        print("  - Computing SAE reconstruction accuracy...")
        try:
            recon_acc_metrics = compute_sae_patched_accuracy(
                model, sae, analysis_dl, act_mean,
                layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE
            )
            wandb.summary["baseline_accuracy"] = recon_acc_metrics["baseline_acc"]
            wandb.summary["sae_patched_task_accuracy"] = recon_acc_metrics["reconstruction_acc"]
            wandb.summary["accuracy_drop"] = recon_acc_metrics["accuracy_drop"]
        except Exception as e:
            print(f"    ⚠ Warning: Could not compute SAE reconstruction accuracy - {e}")
            wandb.summary["sae_patched_task_accuracy"] = 0
            wandb.summary["accuracy_drop"] = 100
        
        # 4. Special features (correlated with attention)
        print("  - Identifying special features...")
        try:
            alpha_d1_all, alpha_d2_all = collect_attention_patterns(
                model, analysis_dl, layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE
            )
            special_features_info = identify_special_features(
                sae_acts_all, alpha_d1_all, alpha_d2_all, threshold=0.5
            )
            
            wandb.summary["n_special_features"] = special_features_info["n_special_features"]
            wandb.summary["special_features_pct"] = 100 * special_features_info["n_special_features"] / d_sae
            wandb.summary["max_attn_correlation"] = float(special_features_info["max_correlation"])
            wandb.summary["mean_abs_attn_correlation"] = float(special_features_info["mean_abs_correlation"])
            
            # Log special features as table
            if special_features_info["special_features"]:
                special_features_table = wandb.Table(
                    columns=["feature_idx", "correlation", "type"],
                    data=[[f["feature_idx"], float(f["correlation"]), f["type"]] 
                          for f in special_features_info["special_features"]]
                )
                wandb.log({"special_features": special_features_table})
                print(f"    ✓ Identified {special_features_info['n_special_features']} special features")
        except Exception as e:
            print(f"    ⚠ Warning: Could not compute special features - {e}")
            wandb.summary["n_special_features"] = NA
        
        print("✓ Logged all analysis metrics to W&B")
        
    except Exception as e:
        print(f"⚠ Error during metric logging: {e}")
        print("Continuing with SAE save...")
        import traceback
        traceback.print_exc()
    
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
