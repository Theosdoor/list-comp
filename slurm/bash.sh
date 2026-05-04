uv sync
source .venv/bin/activate

python scripts/download_wandb_checkpoints.py \
    --entity theo-farrell99-durham-university \
    --project orderbyscale_sae_sweep \
    --artifact_type sae_model \
    --output_dir results/sae_models/
    