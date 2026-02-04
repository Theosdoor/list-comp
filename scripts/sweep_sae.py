"""
W&B Sweep Script for SAE Hyperparameter Search

Runs grid search over SAE configurations with multiple seeds.
Uses scripts/train_sae.py to train each SAE configuration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import subprocess
import wandb


# Base Model Configuration
MODEL_NAME = '2layer_100dig_64d'


def train_sae_sweep():
    """Train SAE with W&B sweep configuration using train_sae.py."""
    
    # Extract hyperparameters from sweep config
    config = wandb.config
    d_sae = config.d_sae
    top_k = config.top_k
    lr = config.lr
    n_steps = config.n_steps
    warmup_steps = config.warmup_steps
    seed = config.seed
    
    # Generate run name matching SAE name format
    run_name = f'sae_d{d_sae}_k{top_k}_lr{lr}_seed{seed}_{MODEL_NAME}'
    
    # Initialize W&B run with custom name
    run = wandb.init(name=run_name)
    
    print(f"\n{'='*60}")
    print(f"Training SAE: {run_name}")
    print(f"{'='*60}\n")
    
    # Build command to call train_sae.py
    train_script = Path(__file__).parent / "train_sae.py"
    cmd = [
        "python3",
        str(train_script),
        "--d_sae", str(d_sae),
        "--top_k", str(top_k),
        "--lr", str(lr),
        "--n_steps", str(n_steps),
        "--warmup_steps", str(warmup_steps),
    ]
    
    # Set environment to pass W&B run info to subprocess
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = run.id
    env["WANDB_RESUME"] = "allow"
    
    # Run training script
    result = subprocess.run(cmd, env=env, capture_output=False)
    
    if result.returncode != 0:
        print(f"✗ Training failed with return code {result.returncode}")
        wandb.finish(exit_code=1)
        raise RuntimeError(f"Training script failed with code {result.returncode}")
    
    print(f"\n✓ Training completed successfully")
    wandb.finish()


if __name__ == "__main__":
    train_sae_sweep()
