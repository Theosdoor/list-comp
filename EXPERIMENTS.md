# Experiments

## 2-Layer Architecture Sweep (2026-04-14)

**Command:**
```bash
wandb sweep sweeps/2layer_flags.yaml   # → ou3mwr1m
wandb sweep sweeps/2layer_dmodel.yaml  # → fvg6y0ad
sbatch slurm/submit_2layer_sweep.sh ou3mwr1m
sbatch slurm/submit_2layer_sweep.sh fvg6y0ad
```

**Output paths:**
- Sweep runs: wandb project `order-by-scale`
- Models (FFFFF d_model=64 only): `models/2_layer_sweep/`
- Results cache: `results/2layer_sweep_cache.csv`

**Config:** 32 flag combos × 30 seeds (d_model=64) + d_model∈{8,32,128} × 30 seeds (FFFFF) = 1050 runs total

**Table generation:**
```bash
python scripts/make_2layer_table.py \
    --flags-sweep-ids rvon63i9 \
    --dmodel-sweep-ids n50fmwsa
```
