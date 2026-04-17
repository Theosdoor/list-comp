# Experiments

## 2-Layer Architecture Sweep (2026-04-14)

**Command:**
```bash
wandb sweep sweeps/2layer_flags.yaml   # → ou3mwr1m
wandb sweep sweeps/2layer_dmodel.yaml  # → fvg6y0ad
sbatch slurm/submit_2layer_sweep.sh ou3mwr1m # DONE!
sbatch slurm/submit_2layer_sweep.sh fvg6y0ad
```

**Output paths:**
- Sweep runs: wandb project `order-by-scale`
- Models (FFFFF d_model=64 only): `models/2_layer_sweep/`
- Results cache: `results/2layer_sweep_cache.csv`

**Config:** 32 flag combos × 30 seeds (d_model=64) + d_model∈{8,32,128} × 30 seeds (FFFFF) = 1050 runs total

**Table generation:**
```bash
python3 scripts/make_2layer_table.py --flags-sweep-ids ou3mwr1m --dmodel-sweep-ids fvg6y0ad --nheads-sweep-ids 7lu56xj8
```


Top 3 fffff models (models/2_layer_sweep/)
% d64_h1_lnF_biasF_wvF_woF_mlpF_s3_acc0.9405.pt   (seed=3, acc=0.9405)
% d64_h1_lnF_biasF_wvF_woF_mlpF_s13_acc0.9343.pt   (seed=13, acc=0.9343)
% d64_h1_lnF_biasF_wvF_woF_mlpF_s9_acc0.9337.pt   (seed=9, acc=0.9337)
