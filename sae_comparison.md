# SAE Sweep Comparison Report

Compared 28 SAE models from sweep runs on 10000 samples.


## Summary Table

| Model | d_sae | k | L0 | Dead | Dead % | Alive | MSE |
|-------|-------|---|----|----|--------|-------|-----|
| sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1781 |
| sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1826 |
| sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1770 |
| sae_d50_k1_lr0.0003_seed42_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1779 |
| sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1831 |
| sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1771 |
| sae_d50_k1_lr0.0004_seed42_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1779 |
| sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1831 |
| sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1783 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 100 | 1 | 0.98 | 1 | 1.0% | 99 | 0.1358 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1401 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1401 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1381 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1395 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1381 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1379 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 50 | 4 | 3.99 | 0 | 0.0% | 50 | 0.1087 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 50 | 4 | 4.02 | 0 | 0.0% | 50 | 0.1079 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 50 | 4 | 3.98 | 0 | 0.0% | 50 | 0.1099 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 50 | 4 | 4.03 | 0 | 0.0% | 50 | 0.1078 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1097 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1076 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1078 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 50 | 4 | 4.02 | 0 | 0.0% | 50 | 0.1097 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1079 |

## Best Models by top_k


### top_k = 1

- Best reconstruction (MSE): **sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d** (MSE: 0.1358, d_sae=100)
- Fewest dead features: **sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

### top_k = 2

- Best reconstruction (MSE): **sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d** (MSE: 0.1379, d_sae=50)
- Fewest dead features: **sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

### top_k = 4

- Best reconstruction (MSE): **sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d** (MSE: 0.1076, d_sae=50)
- Fewest dead features: **sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

## Firing Rate Statistics

| Model | Min Firing | Max Firing | Mean Firing |
|-------|------------|------------|-------------|
| sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d | 0.0132 | 0.0903 | 0.0199 |
| sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d | 0.0151 | 0.0993 | 0.0199 |
| sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d | 0.0127 | 0.0966 | 0.0199 |
| sae_d50_k1_lr0.0003_seed42_2layer_100dig_64d | 0.0128 | 0.0886 | 0.0199 |
| sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d | 0.0152 | 0.1034 | 0.0198 |
| sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d | 0.0125 | 0.0956 | 0.0199 |
| sae_d50_k1_lr0.0004_seed42_2layer_100dig_64d | 0.0128 | 0.0886 | 0.0199 |
| sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d | 0.0152 | 0.1040 | 0.0199 |
| sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d | 0.0125 | 0.0931 | 0.0199 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 0.0011 | 0.0149 | 0.0099 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 0.0198 | 0.3368 | 0.0398 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 0.0199 | 0.4436 | 0.0399 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 0.0197 | 0.4470 | 0.0398 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 0.0199 | 0.3556 | 0.0399 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 0.0199 | 0.4408 | 0.0401 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 0.0195 | 0.4379 | 0.0399 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 0.0199 | 0.3248 | 0.0399 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 0.0199 | 0.4407 | 0.0399 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 0.0197 | 0.4494 | 0.0399 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 0.0199 | 0.5584 | 0.0799 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 0.0207 | 0.5618 | 0.0805 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 0.0199 | 0.5027 | 0.0796 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 0.0207 | 0.5504 | 0.0806 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 0.0207 | 0.5530 | 0.0803 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 0.0214 | 0.5707 | 0.0801 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 0.0200 | 0.5509 | 0.0802 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 0.0206 | 0.5536 | 0.0804 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 0.0208 | 0.5810 | 0.0802 |

## Analysis

- **L0**: Average number of active features per sample (lower = sparser)
- **Dead %**: Percentage of features that never fire (lower = better utilization)
- **MSE**: Mean squared reconstruction error (lower = better reconstruction)
