#!/bin/bash
uv sync
source .venv/bin/activate

python scripts/download_wandb_checkpoints.py xliz4f19
python scripts/download_wandb_checkpoints.py tbxyl1y7
python scripts/download_wandb_checkpoints.py k2bsjr0n
