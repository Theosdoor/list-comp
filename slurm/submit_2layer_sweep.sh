#!/bin/bash
#SBATCH --job-name=2layer_sweep
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=28G

# Usage: sbatch slurm/submit_2layer_sweep.sh <sweep_id>
# Example: sbatch slurm/submit_2layer_sweep.sh <sweep_id>
# Parallel launch example:
#   for id in sweep1 sweep2 sweep3; do sbatch slurm/submit_2layer_sweep.sh $id; done

if [ -z "$1" ]; then
  echo "Error: missing <sweep_id>"
  echo "Usage: sbatch slurm/submit_2layer_sweep.sh <sweep_id>"
  exit 1
fi

SWEEP_ID="$1"

cd /home2/nchw73/Year4/L4_Project/list-comp-priv || { echo "Failed to cd to repo"; exit 1; }

uv sync
source .venv/bin/activate

echo "Node: $(hostname)"
echo "Sweep ID: $SWEEP_ID"

python -c "import torch,subprocess; print('CUDA available:', torch.cuda.is_available());\nif torch.cuda.is_available():\n    print('GPU:', torch.cuda.get_device_name(0))\nelse:\n    try:\n        print('GPU via nvidia-smi:', subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader']).decode().strip())\n    except Exception as e:\n        print('GPU: unknown', e)"

wandb agent theo-farrell99-durham-university/order-by-scale/$SWEEP_ID

echo "Finished at $(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"
