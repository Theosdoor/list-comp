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


# Train validation SAEs (one JumpReLU, one Matryoshka) and run crossover on each
# python3 scripts/nb_train_sae.py --sae_type jumprelu  --d_sae 128 --target_l0 3.0 --n_steps 20000 --lr 7e-5
# python3 scripts/nb_train_sae.py --sae_type matryoshka --d_sae 128 --top_k 3 --n_groups 4 --n_steps 20000 --lr 1e-4

# python3 scripts/run_crossover_analysis.py --sae jumprelu_sae_d128_tl03.0_2layer_100dig_64d.pt --report
# python3 scripts/run_crossover_analysis.py --sae matryoshka_sae_d128_k3_ng4_2layer_100dig_64d.pt --report

python3 scripts/run_crossover_analysis.py --sae sweep_runs/sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt --report

# python3 scripts/nb_compare_sae.py

echo "[slurm] Finished at $(date)"