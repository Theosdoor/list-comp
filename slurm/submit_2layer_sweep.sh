#!/bin/bash
#SBATCH --job-name=2layer_sweep
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --output=slurm/logs/slurm_%j.log

cd /home2/nchw73/Year4/L4_Project/list-comp-priv
uv sync && source .venv/bin/activate
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
wandb agent theo-farrell99-durham-university/order-by-scale/$1
