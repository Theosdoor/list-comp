#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=20G

# go to folder and sync venv
cd "${PROJECT_DIR:-$PWD}"
uv sync
source .venv/bin/activate

# verify we got gpu
echo "[slurm] Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'[slurm] CUDA Available: {torch.cuda.is_available()}'); print(f'[slurm] Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

# python scripts/compare_sae.py --sae-folders sae_checkpoints/ --special-threshold 0.3 --exclude-d-sae 64 100 384 448 512
python scripts/plot_sae_sweep.py --exclude-d-sae 64 100 384 448 512 --exclude-special-col # --exclude-runs-col

# python scripts/special_latents_across_saes.py \
#     --sae_dirs sae_checkpoints/ \
#     --model_path models/2layer_100dig_64d.pt \
#     --alpha_diff_thresh 0.3 \
#     --output_dir results/sae_plots \
#     --exclude_d_sae 64 100 384 448 512 \
#     --exclude_l0 6

echo "[slurm] Finished at $(date)"
