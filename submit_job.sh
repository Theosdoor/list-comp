#!/bin/bash
#SBATCH --job-name=SAEs
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G


# 1. Activate Environment
source /home2/nchw73/venv312/bin/activate

# 2. Go to your folder
cd /home2/nchw73/Year4/L4_Project/list-comp-priv/

# 3. Debug: Verify we got the GPU
echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

# 4. Run the experiments

python model_scripts/train.py

# # Sweep over dictionary sizes
# python train_btk_sae.py --d_sae 100
# python train_btk_sae.py --d_sae 150
# python train_btk_sae.py --d_sae 200
# python train_btk_sae.py --d_sae 256

# # Or try different k values
# python train_btk_sae.py --d_sae 150 --top_k 2
# python train_btk_sae.py --d_sae 150 --top_k 4
# python train_btk_sae.py --d_sae 150 --top_k 8
