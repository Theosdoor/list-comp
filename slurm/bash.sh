uv sync
source .venv/bin/activate

python scripts/download_wandb_checkpoints.py \
    --project <entity>/<project> \
    --artifact_type sae_model \
    --output_dir results/sae_models/
    
