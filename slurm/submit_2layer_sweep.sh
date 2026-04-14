#!/bin/bash
#SBATCH --job-name=2layer_sweep
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=28G
set -euo pipefail

# Usage: sbatch slurm/submit_2layer_sweep.sh <sweep_id>
# Example: sbatch slurm/submit_2layer_sweep.sh <sweep_id>
# Parallel launch example (repeat submissions for same sweep_id):
#   for i in 1 2 3; do sbatch slurm/submit_2layer_sweep.sh <sweep_id>; done

if [ -z "${1:-}" ]; then
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

python3 - <<'PY'
import torch
if torch.cuda.is_available():
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "Unknown CUDA device"
    print(f'[slurm] CUDA Available: True')
    print(f'[slurm] Device: {name}')
else:
    print(f'[slurm] CUDA Available: False')
    print(f'[slurm] Device: CPU')
PY

wandb agent theo-farrell99-durham-university/order-by-scale/$SWEEP_ID

echo "Finished at $(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"
