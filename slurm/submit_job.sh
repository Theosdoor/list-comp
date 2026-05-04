#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=20G

# go to folder and sync venv
cd /home2/nchw73/Year4/L4_Project/list-comp # NOT priv
uv sync
source .venv/bin/activate

# verify we got gpu
echo "[slurm] Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'[slurm] CUDA Available: {torch.cuda.is_available()}'); print(f'[slurm] Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

# python scripts/special_latents_across_saes.py \
#     --sae_dirs sae_checkpoints/ \
#     --model_path models/2layer_100dig_64d.pt \
#     --alpha_diff_thresh 0.5 \
#     --output_dir results/sae_plots

# python scripts/plot_sae_analysis.py \
#     --sae_dirs sae_checkpoints/ \
#     --model_path models/2layer_100dig_64d.pt \
#     --alpha_diff_thresh 0.5 \
#     --output_dir results/sae_plots \
#     --exclude_l0 1 2

# python scripts/compare_sae.py --sae_folders sae_checkpoints/
python scripts/plot_sae_sweep.py --exclude-d-sae 64 100 384 448 512 --exclude-runs-col --exclude-special-col

# python3 visualisation/special_latents_across_saes.py \
#    --local-folder sae_checkpoints/sweep_tbxyl1y7 \
#    --threshold 0.3

# python3 visualisation/special_latents_across_saes.py \
#    --local-folder sae_checkpoints/sweep_xliz4f19 \
#    --threshold 0.3

# python3 visualisation/special_latents_across_saes.py \
#    --local-folder sae_checkpoints/sweep_k2bsjr0n \
#    --threshold 0.3

# python3 visualisation/special_latents_across_saes.py \
#    --local-folder sae_checkpoints/sweep_k2bsjr0n sae_checkpoints/sweep_xliz4f19 sae_checkpoints/sweep_tbxyl1y7\
#    --threshold 0.3

echo "[slurm] Finished at $(date)"