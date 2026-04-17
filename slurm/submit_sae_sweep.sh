#!/bin/bash
#SBATCH --job-name=SAEs
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=48:00:00
#SBATCH --mem=28G

# go to folder and sync venv
cd /home2/nchw73/Year4/L4_Project/list-comp-priv
uv sync
source .venv/bin/activate

# verify we got gpu
echo "[slurm] Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'[slurm] CUDA Available: {torch.cuda.is_available()}'); print(f'[slurm] Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"


# btk
wandb agent theo-farrell99-durham-university/orderbyscale_sae_sweep/5awoztdt

# matryoshka
# wandb agent theo-farrell99-durham-university/orderbyscale_sae_sweep/tbxyl1y7

# jumprelu
# wandb agent theo-farrell99-durham-university/orderbyscale_sae_sweep/xliz4f19