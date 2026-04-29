#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=20G

# go to folder and sync venv
cd /home2/nchw73/Year4/L4_Project/list-comp-priv
uv sync
source .venv/bin/activate

# verify we got gpu
echo "[slurm] Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'[slurm] CUDA Available: {torch.cuda.is_available()}'); print(f'[slurm] Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

# python visualization/compare_sae.py --sae_folders results/sae_models/sweep_k2bsjr0n results/sae_models/sweep_tbxyl1y7 results/sae_models/sweep_xliz4f19
# python scripts/plot_sae_sweep.py --exclude-d-sae 100 448 512 --exclude-runs-col --exclude-special-col

python3 scripts/compute_sae_metrics.py \
   --sae-folder results/sae_models \
   --thresholds 0.1 0.3 0.5 0.8

python3 visualisation/special_latents_across_saes.py \
   --local-folder results/sae_models/sweep_tbxyl1y7 \
   --threshold 0.3

python3 visualisation/special_latents_across_saes.py \
   --local-folder results/sae_models/sweep_xliz4f19 \
   --threshold 0.3

python3 visualisation/special_latents_across_saes.py \
   --local-folder results/sae_models/sweep_k2bsjr0n \
   --threshold 0.3

python3 visualisation/special_latents_across_saes.py \
   --local-folder results/sae_models/sweep_k2bsjr0n results/sae_models/sweep_xliz4f19 results/sae_models/sweep_tbxyl1y7\
   --threshold 0.3

echo "[slurm] Finished at $(date)"