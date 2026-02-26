#!/bin/bash
#SBATCH --job-name=SAEs
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=28G

# 1. Activate Environment
cd /home2/nchw73/Year4/L4_Project/list-comp-priv/
uv sync
source .venv/bin/activate

echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

# run
# python3 scripts/run_crossover_analysis.py
# python3 scripts/analyze_failure_reasons.py


# SAE sweep (comment/uncomment as needed)
wandb agent theo-farrell99-durham-university/btksae_sweep/gw95m2e2
python3 scripts/compare_sae.py
