#!/bin/bash
#SBATCH --job-name=xovers
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

# jumprelu 256/4 - features 195, 175, 211
python3 scripts/run_crossover_analysis.py \
   --model 2layer_100dig_64d \
   --sae results/sae_models/sweep_xliz4f19/jumprelu_sae_d256_tl05_sp5_lr7e-05_seed1_2layer_100dig_64d.pt \
   --feature auto \
   --threshold 0.3 \
   --report

# jumprelu 128/3
python3 scripts/run_crossover_analysis.py \
   --model 2layer_100dig_64d \
   --sae results/sae_models/sweep_xliz4f19/jumprelu_sae_d128_tl05_sp5_lr7e-05_seed4_2layer_100dig_64d.pt \
   --feature auto \
   --threshold 0.3 \
   --report

# matry 256/3
python3 scripts/run_crossover_analysis.py \
   --model 2layer_100dig_64d \
   --sae results/sae_models/sweep_tbxyl1y7/matryoshka_sae_d256_k3_ng2_lr0.0003_seed4_2layer_100dig_64d.pt \
   --feature auto \
   --threshold 0.3 \
   --report

# matry 192/4
python3 scripts/run_crossover_analysis.py \
   --model 2layer_100dig_64d \
   --sae results/sae_models/sweep_tbxyl1y7/matryoshka_sae_d192_k4_ng4_lr0.0003_seed0_2layer_100dig_64d.pt \
   --feature auto \
   --threshold 0.3 \
   --report

# btk_sae_d192_k3_lr0.0003_seed5 
python3 scripts/run_crossover_analysis.py \
   --model 2layer_100dig_64d \
   --sae results/sae_models/sweep_k2bsjr0n/btk_sae_d192_k3_lr0.0003_seed5_2layer_100dig_64d.pt \
   --feature auto \
   --threshold 0.3 \
   --report

# btk_sae_d128_k5_lr0.0003_seed0
python3 scripts/run_crossover_analysis.py \
   --model 2layer_100dig_64d \
   --sae results/sae_models/sweep_k2bsjr0n/btk_sae_d128_k5_lr0.0003_seed0_2layer_100dig_64d.pt \
   --feature auto \
   --threshold 0.3 \
   --report

# (better) btk 128/3 - feats 29, 102
# python3 scripts/run_crossover_analysis.py \
#    --model 2layer_100dig_64d \
#    --sae results/sae_models/sweep_k2bsjr0n/btk_sae_d128_k3_lr0.0003_seed3_2layer_100dig_64d.pt \
#    --feature auto \
#    --threshold 0.5 \
#    --report

echo "[slurm] Finished at $(date)"