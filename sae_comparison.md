# SAE Sweep Comparison Report

Compared 77 SAE models from sweep runs on 10000 samples.


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
| sae_d50_k1_lr0.001_seed42_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1778 |
| sae_d50_k1_lr0.001_seed43_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1831 |
| sae_d50_k1_lr0.001_seed44_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1783 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 100 | 1 | 0.98 | 1 | 1.0% | 99 | 0.1358 |
| sae_d100_k1_lr0.0001_seed43_2layer_100dig_64d | 100 | 1 | 0.98 | 1 | 1.0% | 99 | 0.1357 |
| sae_d100_k1_lr0.0001_seed44_2layer_100dig_64d | 100 | 1 | 0.98 | 2 | 2.0% | 98 | 0.1355 |
| sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d | 100 | 1 | 0.98 | 2 | 2.0% | 98 | 0.1347 |
| sae_d100_k1_lr0.0003_seed43_2layer_100dig_64d | 100 | 1 | 0.98 | 1 | 1.0% | 99 | 0.1359 |
| sae_d100_k1_lr0.0003_seed44_2layer_100dig_64d | 100 | 1 | 0.98 | 0 | 0.0% | 100 | 0.1350 |
| sae_d100_k1_lr0.001_seed42_2layer_100dig_64d | 100 | 1 | 0.98 | 2 | 2.0% | 98 | 0.1352 |
| sae_d100_k1_lr0.001_seed43_2layer_100dig_64d | 100 | 1 | 0.98 | 0 | 0.0% | 100 | 0.1365 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1401 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1401 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1381 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1395 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1381 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1379 |
| sae_d50_k2_lr0.001_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1399 |
| sae_d50_k2_lr0.001_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1397 |
| sae_d50_k2_lr0.001_seed44_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1378 |
| sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d | 100 | 2 | 2.00 | 1 | 1.0% | 99 | 0.0291 |
| sae_d100_k2_lr0.0001_seed43_2layer_100dig_64d | 100 | 2 | 1.98 | 1 | 1.0% | 99 | 0.0317 |
| sae_d100_k2_lr0.0001_seed44_2layer_100dig_64d | 100 | 2 | 2.29 | 0 | 0.0% | 100 | 0.0540 |
| sae_d100_k2_lr0.0003_seed42_2layer_100dig_64d | 100 | 2 | 1.98 | 1 | 1.0% | 99 | 0.0296 |
| sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d | 100 | 2 | 1.99 | 1 | 1.0% | 99 | 0.0296 |
| sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d | 100 | 2 | 1.94 | 0 | 0.0% | 100 | 0.0246 |
| sae_d100_k2_lr0.001_seed42_2layer_100dig_64d | 100 | 2 | 2.00 | 0 | 0.0% | 100 | 0.0264 |
| sae_d100_k2_lr0.001_seed43_2layer_100dig_64d | 100 | 2 | 2.00 | 1 | 1.0% | 99 | 0.0293 |
| sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1210 |
| sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1233 |
| sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1209 |
| sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d | 50 | 3 | 3.01 | 0 | 0.0% | 50 | 0.1224 |
| sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d | 50 | 3 | 3.01 | 0 | 0.0% | 50 | 0.1233 |
| sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1203 |
| sae_d50_k3_lr0.001_seed42_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1203 |
| sae_d50_k3_lr0.001_seed43_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1216 |
| sae_d50_k3_lr0.001_seed44_2layer_100dig_64d | 50 | 3 | 3.01 | 0 | 0.0% | 50 | 0.1198 |
| sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d | 100 | 3 | 2.94 | 1 | 1.0% | 99 | 0.0052 |
| sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d | 100 | 3 | 2.86 | 0 | 0.0% | 100 | 0.0036 |
| sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d | 100 | 3 | 2.84 | 0 | 0.0% | 100 | 0.0039 |
| sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d | 100 | 3 | 3.05 | 0 | 0.0% | 100 | 0.0045 |
| sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d | 100 | 3 | 2.92 | 0 | 0.0% | 100 | 0.0038 |
| sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d | 100 | 3 | 3.02 | 0 | 0.0% | 100 | 0.0042 |
| sae_d100_k3_lr0.001_seed42_2layer_100dig_64d | 100 | 3 | 3.00 | 1 | 1.0% | 99 | 0.0055 |
| sae_d100_k3_lr0.001_seed43_2layer_100dig_64d | 100 | 3 | 2.98 | 0 | 0.0% | 100 | 0.0037 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 50 | 4 | 3.99 | 0 | 0.0% | 50 | 0.1087 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 50 | 4 | 4.02 | 0 | 0.0% | 50 | 0.1079 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 50 | 4 | 3.98 | 0 | 0.0% | 50 | 0.1099 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 50 | 4 | 4.03 | 0 | 0.0% | 50 | 0.1078 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1097 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1076 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1078 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 50 | 4 | 4.02 | 0 | 0.0% | 50 | 0.1097 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1079 |
| sae_d50_k4_lr0.001_seed42_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1076 |
| sae_d50_k4_lr0.001_seed43_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1077 |
| sae_d50_k4_lr0.001_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1069 |
| sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d | 100 | 4 | 4.05 | 0 | 0.0% | 100 | 0.0044 |
| sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d | 100 | 4 | 4.10 | 0 | 0.0% | 100 | 0.0047 |
| sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d | 100 | 4 | 3.94 | 0 | 0.0% | 100 | 0.0041 |
| sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d | 100 | 4 | 4.09 | 0 | 0.0% | 100 | 0.0047 |
| sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d | 100 | 4 | 3.95 | 0 | 0.0% | 100 | 0.0048 |
| sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d | 100 | 4 | 4.06 | 0 | 0.0% | 100 | 0.0048 |
| sae_d100_k4_lr0.001_seed42_2layer_100dig_64d | 100 | 4 | 3.97 | 0 | 0.0% | 100 | 0.0048 |
| sae_d100_k4_lr0.001_seed43_2layer_100dig_64d | 100 | 4 | 3.97 | 1 | 1.0% | 99 | 0.0054 |

## Best Models by top_k


### top_k = 1

- Best reconstruction (MSE): **sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d** (MSE: 0.1347, d_sae=100)
- Fewest dead features: **sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

### top_k = 2

- Best reconstruction (MSE): **sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d** (MSE: 0.0246, d_sae=100)
- Fewest dead features: **sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

### top_k = 3

- Best reconstruction (MSE): **sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d** (MSE: 0.0036, d_sae=100)
- Fewest dead features: **sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

### top_k = 4

- Best reconstruction (MSE): **sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d** (MSE: 0.0041, d_sae=100)
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
| sae_d50_k1_lr0.001_seed42_2layer_100dig_64d | 0.0133 | 0.0658 | 0.0199 |
| sae_d50_k1_lr0.001_seed43_2layer_100dig_64d | 0.0150 | 0.0984 | 0.0198 |
| sae_d50_k1_lr0.001_seed44_2layer_100dig_64d | 0.0123 | 0.0940 | 0.0198 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 0.0011 | 0.0149 | 0.0099 |
| sae_d100_k1_lr0.0001_seed43_2layer_100dig_64d | 0.0024 | 0.0242 | 0.0099 |
| sae_d100_k1_lr0.0001_seed44_2layer_100dig_64d | 0.0025 | 0.0141 | 0.0100 |
| sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d | 0.0013 | 0.0149 | 0.0100 |
| sae_d100_k1_lr0.0003_seed43_2layer_100dig_64d | 0.0025 | 0.0242 | 0.0099 |
| sae_d100_k1_lr0.0003_seed44_2layer_100dig_64d | 0.0034 | 0.0190 | 0.0098 |
| sae_d100_k1_lr0.001_seed42_2layer_100dig_64d | 0.0030 | 0.0167 | 0.0100 |
| sae_d100_k1_lr0.001_seed43_2layer_100dig_64d | 0.0004 | 0.0225 | 0.0098 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 0.0198 | 0.3368 | 0.0398 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 0.0199 | 0.4436 | 0.0399 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 0.0197 | 0.4470 | 0.0398 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 0.0199 | 0.3556 | 0.0399 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 0.0199 | 0.4408 | 0.0401 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 0.0195 | 0.4379 | 0.0399 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 0.0199 | 0.3248 | 0.0399 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 0.0199 | 0.4407 | 0.0399 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 0.0197 | 0.4494 | 0.0399 |
| sae_d50_k2_lr0.001_seed42_2layer_100dig_64d | 0.0198 | 0.3354 | 0.0399 |
| sae_d50_k2_lr0.001_seed43_2layer_100dig_64d | 0.0199 | 0.4439 | 0.0401 |
| sae_d50_k2_lr0.001_seed44_2layer_100dig_64d | 0.0196 | 0.4237 | 0.0399 |
| sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d | 0.0013 | 0.1735 | 0.0202 |
| sae_d100_k2_lr0.0001_seed43_2layer_100dig_64d | 0.0078 | 0.0944 | 0.0200 |
| sae_d100_k2_lr0.0001_seed44_2layer_100dig_64d | 0.0001 | 0.0607 | 0.0229 |
| sae_d100_k2_lr0.0003_seed42_2layer_100dig_64d | 0.0003 | 0.0662 | 0.0200 |
| sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d | 0.0084 | 0.1171 | 0.0201 |
| sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d | 0.0088 | 0.0442 | 0.0194 |
| sae_d100_k2_lr0.001_seed42_2layer_100dig_64d | 0.0001 | 0.0653 | 0.0200 |
| sae_d100_k2_lr0.001_seed43_2layer_100dig_64d | 0.0053 | 0.0633 | 0.0202 |
| sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d | 0.0199 | 0.4971 | 0.0600 |
| sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d | 0.0199 | 0.4277 | 0.0601 |
| sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d | 0.0198 | 0.5037 | 0.0601 |
| sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d | 0.0206 | 0.5011 | 0.0601 |
| sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d | 0.0203 | 0.4259 | 0.0602 |
| sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d | 0.0199 | 0.4999 | 0.0601 |
| sae_d50_k3_lr0.001_seed42_2layer_100dig_64d | 0.0199 | 0.5173 | 0.0600 |
| sae_d50_k3_lr0.001_seed43_2layer_100dig_64d | 0.0200 | 0.4864 | 0.0600 |
| sae_d50_k3_lr0.001_seed44_2layer_100dig_64d | 0.0199 | 0.5039 | 0.0601 |
| sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d | 0.0199 | 0.7498 | 0.0297 |
| sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d | 0.0198 | 0.7123 | 0.0286 |
| sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d | 0.0197 | 0.7131 | 0.0284 |
| sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d | 0.0199 | 0.7345 | 0.0305 |
| sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d | 0.0197 | 0.6989 | 0.0292 |
| sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d | 0.0199 | 0.7054 | 0.0302 |
| sae_d100_k3_lr0.001_seed42_2layer_100dig_64d | 0.0199 | 0.7132 | 0.0303 |
| sae_d100_k3_lr0.001_seed43_2layer_100dig_64d | 0.0199 | 0.7241 | 0.0298 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 0.0199 | 0.5584 | 0.0799 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 0.0207 | 0.5618 | 0.0805 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 0.0199 | 0.5027 | 0.0796 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 0.0207 | 0.5504 | 0.0806 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 0.0207 | 0.5530 | 0.0803 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 0.0214 | 0.5707 | 0.0801 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 0.0200 | 0.5509 | 0.0802 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 0.0206 | 0.5536 | 0.0804 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 0.0208 | 0.5810 | 0.0802 |
| sae_d50_k4_lr0.001_seed42_2layer_100dig_64d | 0.0207 | 0.5640 | 0.0802 |
| sae_d50_k4_lr0.001_seed43_2layer_100dig_64d | 0.0209 | 0.5621 | 0.0802 |
| sae_d50_k4_lr0.001_seed44_2layer_100dig_64d | 0.0204 | 0.5582 | 0.0802 |
| sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d | 0.0204 | 0.7053 | 0.0405 |
| sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d | 0.0200 | 0.5077 | 0.0410 |
| sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d | 0.0200 | 0.8354 | 0.0394 |
| sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d | 0.0201 | 0.6384 | 0.0409 |
| sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d | 0.0200 | 0.4710 | 0.0395 |
| sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d | 0.0199 | 0.6538 | 0.0406 |
| sae_d100_k4_lr0.001_seed42_2layer_100dig_64d | 0.0199 | 0.6663 | 0.0397 |
| sae_d100_k4_lr0.001_seed43_2layer_100dig_64d | 0.0199 | 0.4792 | 0.0401 |

## Analysis

- **L0**: Average number of active features per sample (lower = sparser)
- **Dead %**: Percentage of features that never fire (lower = better utilization)
- **MSE**: Mean squared reconstruction error (lower = better reconstruction)
