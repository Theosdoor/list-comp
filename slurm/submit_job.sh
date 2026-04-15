#!/bin/bash
#SBATCH --job-name=SAEs
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
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


# Run the experiments
# python3 scripts/compare_sae.py

SAE="sweep_runs/sae_d192_k3_lr0.0001_seed2_2layer_100dig_64d.pt"
# python3 scripts/run_crossover_analysis.py --sae "$SAE"
python3 scripts/analyze_failure_reasons.py --sae "$SAE"

# SAE sweep (comment/uncomment as needed)
# wandb agent theo-farrell99-durham-university/btksae_sweep/x7tgo6fv

echo "[slurm] Finished at $(date)"