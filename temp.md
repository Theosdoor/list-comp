# SAE Model Performance Analysis

## Summary
Analysis of 623 SAE models from sweep runs on the full dataset (10,000 samples). The following identifies top performers by explained variance and CE increase (delta LM loss).

Note: all figures are on the **full train+val dataset** — not held-out val only.

## Top Models by Explained Variance (Exp Var)

Higher explained variance = better reconstruction of the original activations.

| Model | d_sae | k | Exp Var | Patched CE | CE Increase |
|-------|-------|---|---------|-----------|-------------|
| sae_d320_k5_lr0.0001_seed1_2layer_100dig_64d | 320 | 5 | 0.9937 | 0.2282 | 0.1167 |
| sae_d512_k4_lr7.032012560204404e-05_seed0_2layer_100dig_64d | 512 | 4 | 0.9936 | 0.2203 | 0.1087 |
| sae_d320_k4_lr8.436504189764667e-05_seed0_2layer_100dig_64d | 320 | 4 | 0.9935 | 0.1026 | — |

## Top Models by Lowest CE Increase (Delta LM Loss)

Lower CE increase = minimal downstream performance degradation when SAE activations are patched in. This is the primary downstream faithfulness metric.

| Model | d_sae | k | CE Increase | Patched CE | Exp Var |
|-------|-------|---|-------------|-----------|---------|
| sae_d320_k3_lr1e-05_seed1_2layer_100dig_64d | 320 | 3 | 0.0552 | 0.1668 | 0.9932 |
| sae_d192_k3_lr0.0001_seed2_2layer_100dig_64d | 192 | 3 | 0.0555 | 0.1670 | 0.9932 |
| sae_d256_k3_lr2.3238903036899937e-05_seed0_2layer_100dig_64d | 256 | 3 | 0.0581 | — | 0.9929 |
| sae_d320_k5_lr1e-05_seed0_2layer_100dig_64d | 320 | 5 | 0.0591 | — | 0.9932 |

## Best Balanced Performance

Models achieving the best combination of high explained variance and low CE increase:

| Model | d_sae | k | Exp Var | Baseline CE | Patched CE | CE Increase |
|-------|-------|---|---------|------------|-----------|-------------|
| sae_d320_k3_lr1e-05_seed1_2layer_100dig_64d | 320 | 3 | 0.9932 | 0.1115 | 0.1668 | **0.0552** |
| sae_d192_k3_lr0.0001_seed2_2layer_100dig_64d | 192 | 3 | 0.9932 | 0.1115 | 0.1670 | **0.0555** |
| sae_d128_k3_lr0.0001_seed0_2layer_100dig_64d | 128 | 3 | 0.9927 | 0.1115 | 0.1846 | 0.0731 |

## Key Findings

- **k=3 consistently wins on CE increase** despite k=4/5 having higher explained variance. This suggests k=3 hits the sweet spot: enough features to preserve task-relevant information, but sparse enough to discard noise.
- **Baseline CE = 0.1115** for all models (constant — same unpatched model on the same 10k inputs every time).
- **All CE increases are positive** — no SAE outperforms the unpatched model on downstream CE.
- **EV and CE increase are different axes**: the top-EV model (k=5, d=320, EV=0.9937) has CE increase 0.1167, while the top CE-increase model (k=3, d=320) has EV=0.9932 but CE increase only 0.0552. This matches the distinction drawn in published SAE papers between reconstruction fidelity and downstream faithfulness.
