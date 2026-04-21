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

## n_layers × list_len Grid Sweep (2026-04-21)

**Sweep ID:** `2nu3lkwf` (wandb project `order-by-scale`)

**Config:** `sweeps/listlen_nlayers.yaml` — list_len∈{1..10}, n_layers∈{1..10}, 30 seeds, d_model=64, n_heads=1, FFFFF flags, n_digits=100

**Problem discovered:** L4+ runs were training at <15% accuracy after 8+ hours.
Root cause: `get_dataset` enumerates *all* `n_digits^list_len` combinations.
- L3: 100^3 = 1M rows → fast; 50k steps ≈ 128 epochs → converges in ~30 min
- L4: 100^4 = 100M rows → ~14 GB RAM; 50k steps < 1 epoch → cannot converge

**Fix (2026-04-21):** Added `MAX_DATASET_SIZE = 1_000_000` cap in `src/data/datasets.py`.
When `n_digits^list_len > MAX_DATASET_SIZE`, the dataset randomly samples 1M sequences
via `torch.randint` instead of enumerating all combinations. L2/L3 behaviour is unchanged
(their full datasets are ≤1M rows). The list-copy task generalises from a random sample
because digit identity is arbitrary — the model just needs to see enough diversity.
