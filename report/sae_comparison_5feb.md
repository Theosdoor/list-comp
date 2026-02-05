# SAE Sweep Comparison Report

Compared 79 SAE models from sweep runs on 10000 samples.


## Summary Table

| Model | d_sae | k | L0 | Dead | Dead % | Alive | MSE | Exp Var | Recon Acc | Acc Drop |
|-------|-------|---|----|----|--------|-------|-----|---------|-----------|----------|
| sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1781 | 0.5461 | 0.3229 | 0.6321 |
| sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1826 | 0.5347 | 0.3248 | 0.6301 |
| sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1770 | 0.5490 | 0.3217 | 0.6332 |
| sae_d50_k1_lr0.0003_seed42_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1779 | 0.5466 | 0.3174 | 0.6376 |
| sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1831 | 0.5333 | 0.3212 | 0.6337 |
| sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1771 | 0.5487 | 0.3207 | 0.6342 |
| sae_d50_k1_lr0.0004_seed42_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1779 | 0.5466 | 0.3174 | 0.6375 |
| sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1831 | 0.5335 | 0.3215 | 0.6334 |
| sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1783 | 0.5457 | 0.3214 | 0.6335 |
| sae_d50_k1_lr0.001_seed42_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1778 | 0.5470 | 0.3200 | 0.6349 |
| sae_d50_k1_lr0.001_seed43_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1831 | 0.5333 | 0.3211 | 0.6338 |
| sae_d50_k1_lr0.001_seed44_2layer_100dig_64d | 50 | 1 | 0.99 | 0 | 0.0% | 50 | 0.1783 | 0.5456 | 0.3213 | 0.6337 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 100 | 1 | 0.98 | 1 | 1.0% | 99 | 0.1358 | 0.6540 | 0.3841 | 0.5708 |
| sae_d100_k1_lr0.0001_seed43_2layer_100dig_64d | 100 | 1 | 0.98 | 1 | 1.0% | 99 | 0.1357 | 0.6541 | 0.3851 | 0.5699 |
| sae_d100_k1_lr0.0001_seed44_2layer_100dig_64d | 100 | 1 | 0.98 | 2 | 2.0% | 98 | 0.1355 | 0.6548 | 0.3920 | 0.5630 |
| sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d | 100 | 1 | 0.98 | 2 | 2.0% | 98 | 0.1347 | 0.6566 | 0.3886 | 0.5663 |
| sae_d100_k1_lr0.0003_seed43_2layer_100dig_64d | 100 | 1 | 0.98 | 1 | 1.0% | 99 | 0.1359 | 0.6536 | 0.3883 | 0.5666 |
| sae_d100_k1_lr0.0003_seed44_2layer_100dig_64d | 100 | 1 | 0.98 | 0 | 0.0% | 100 | 0.1350 | 0.6559 | 0.3890 | 0.5659 |
| sae_d100_k1_lr0.001_seed42_2layer_100dig_64d | 100 | 1 | 0.98 | 2 | 2.0% | 98 | 0.1352 | 0.6555 | 0.3915 | 0.5635 |
| sae_d100_k1_lr0.001_seed43_2layer_100dig_64d | 100 | 1 | 0.98 | 0 | 0.0% | 100 | 0.1365 | 0.6522 | 0.3868 | 0.5681 |
| sae_d256_k1_2layer_100dig_64d | 256 | 1 | 0.99 | 121 | 47.3% | 135 | 0.1303 | 0.6679 | 0.4033 | 0.5516 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1401 | 0.6429 | 0.4602 | 0.4948 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1401 | 0.6431 | 0.4792 | 0.4758 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1381 | 0.6480 | 0.4825 | 0.4724 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1395 | 0.6444 | 0.4587 | 0.4963 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 | 0.6436 | 0.4743 | 0.4807 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1381 | 0.6481 | 0.4800 | 0.4749 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 | 0.6435 | 0.4610 | 0.4940 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1399 | 0.6435 | 0.4743 | 0.4806 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1379 | 0.6485 | 0.4814 | 0.4735 |
| sae_d50_k2_lr0.001_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1399 | 0.6435 | 0.4658 | 0.4891 |
| sae_d50_k2_lr0.001_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1397 | 0.6439 | 0.4761 | 0.4788 |
| sae_d50_k2_lr0.001_seed44_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1378 | 0.6488 | 0.4728 | 0.4821 |
| sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d | 100 | 2 | 2.00 | 1 | 1.0% | 99 | 0.0291 | 0.9259 | 0.7102 | 0.2447 |
| sae_d100_k2_lr0.0001_seed43_2layer_100dig_64d | 100 | 2 | 1.98 | 1 | 1.0% | 99 | 0.0317 | 0.9192 | 0.6981 | 0.2568 |
| sae_d100_k2_lr0.0001_seed44_2layer_100dig_64d | 100 | 2 | 2.29 | 0 | 0.0% | 100 | 0.0540 | 0.8624 | 0.7281 | 0.2268 |
| sae_d100_k2_lr0.0003_seed42_2layer_100dig_64d | 100 | 2 | 1.98 | 1 | 1.0% | 99 | 0.0296 | 0.9245 | 0.7218 | 0.2332 |
| sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d | 100 | 2 | 1.99 | 1 | 1.0% | 99 | 0.0296 | 0.9245 | 0.7262 | 0.2287 |
| sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d | 100 | 2 | 1.94 | 0 | 0.0% | 100 | 0.0246 | 0.9374 | 0.7365 | 0.2185 |
| sae_d100_k2_lr0.001_seed42_2layer_100dig_64d | 100 | 2 | 2.00 | 0 | 0.0% | 100 | 0.0264 | 0.9328 | 0.7662 | 0.1887 |
| sae_d100_k2_lr0.001_seed43_2layer_100dig_64d | 100 | 2 | 2.00 | 1 | 1.0% | 99 | 0.0293 | 0.9254 | 0.7578 | 0.1972 |
| sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1210 | 0.6918 | 0.5640 | 0.3910 |
| sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1233 | 0.6859 | 0.5655 | 0.3894 |
| sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1209 | 0.6919 | 0.5686 | 0.3863 |
| sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d | 50 | 3 | 3.01 | 0 | 0.0% | 50 | 0.1224 | 0.6880 | 0.5641 | 0.3908 |
| sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d | 50 | 3 | 3.01 | 0 | 0.0% | 50 | 0.1233 | 0.6857 | 0.5655 | 0.3894 |
| sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1203 | 0.6934 | 0.5672 | 0.3878 |
| sae_d50_k3_lr0.001_seed42_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1203 | 0.6933 | 0.5683 | 0.3867 |
| sae_d50_k3_lr0.001_seed43_2layer_100dig_64d | 50 | 3 | 3.00 | 0 | 0.0% | 50 | 0.1216 | 0.6900 | 0.5737 | 0.3812 |
| sae_d50_k3_lr0.001_seed44_2layer_100dig_64d | 50 | 3 | 3.01 | 0 | 0.0% | 50 | 0.1198 | 0.6947 | 0.5674 | 0.3876 |
| sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d | 100 | 3 | 2.94 | 1 | 1.0% | 99 | 0.0052 | 0.9866 | 0.9257 | 0.0292 |
| sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d | 100 | 3 | 2.86 | 0 | 0.0% | 100 | 0.0036 | 0.9909 | 0.9344 | 0.0206 |
| sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d | 100 | 3 | 2.84 | 0 | 0.0% | 100 | 0.0039 | 0.9900 | 0.9304 | 0.0245 |
| sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d | 100 | 3 | 3.05 | 0 | 0.0% | 100 | 0.0045 | 0.9886 | 0.9290 | 0.0259 |
| sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d | 100 | 3 | 2.92 | 0 | 0.0% | 100 | 0.0038 | 0.9904 | 0.9335 | 0.0214 |
| sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d | 100 | 3 | 3.02 | 0 | 0.0% | 100 | 0.0042 | 0.9894 | 0.9312 | 0.0237 |
| sae_d100_k3_lr0.001_seed42_2layer_100dig_64d | 100 | 3 | 3.00 | 1 | 1.0% | 99 | 0.0055 | 0.9860 | 0.9271 | 0.0278 |
| sae_d100_k3_lr0.001_seed43_2layer_100dig_64d | 100 | 3 | 2.98 | 0 | 0.0% | 100 | 0.0037 | 0.9905 | 0.9337 | 0.0212 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 50 | 4 | 3.99 | 0 | 0.0% | 50 | 0.1087 | 0.7229 | 0.6399 | 0.3150 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 50 | 4 | 4.02 | 0 | 0.0% | 50 | 0.1079 | 0.7251 | 0.6664 | 0.2885 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 50 | 4 | 3.98 | 0 | 0.0% | 50 | 0.1099 | 0.7199 | 0.6681 | 0.2868 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 50 | 4 | 4.03 | 0 | 0.0% | 50 | 0.1078 | 0.7252 | 0.6453 | 0.3096 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1097 | 0.7205 | 0.6844 | 0.2705 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1076 | 0.7257 | 0.6593 | 0.2956 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1078 | 0.7253 | 0.6371 | 0.3178 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 50 | 4 | 4.02 | 0 | 0.0% | 50 | 0.1097 | 0.7205 | 0.6848 | 0.2702 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1079 | 0.7250 | 0.6568 | 0.2981 |
| sae_d50_k4_lr0.001_seed42_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1076 | 0.7257 | 0.6591 | 0.2959 |
| sae_d50_k4_lr0.001_seed43_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1077 | 0.7255 | 0.6769 | 0.2781 |
| sae_d50_k4_lr0.001_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1069 | 0.7275 | 0.6580 | 0.2969 |
| sae_d100_k4_50ksteps_2layer_100dig_64d | 100 | 4 | 3.89 | 0 | 0.0% | 100 | 0.0045 | 0.9885 | 0.9316 | 0.0233 |
| sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d | 100 | 4 | 4.05 | 0 | 0.0% | 100 | 0.0044 | 0.9889 | 0.9310 | 0.0240 |
| sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d | 100 | 4 | 4.10 | 0 | 0.0% | 100 | 0.0047 | 0.9880 | 0.9321 | 0.0228 |
| sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d | 100 | 4 | 3.94 | 0 | 0.0% | 100 | 0.0041 | 0.9897 | 0.9330 | 0.0219 |
| sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d | 100 | 4 | 4.09 | 0 | 0.0% | 100 | 0.0047 | 0.9880 | 0.9267 | 0.0282 |
| sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d | 100 | 4 | 3.95 | 0 | 0.0% | 100 | 0.0048 | 0.9877 | 0.9271 | 0.0279 |
| sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d | 100 | 4 | 4.06 | 0 | 0.0% | 100 | 0.0048 | 0.9877 | 0.9297 | 0.0252 |
| sae_d100_k4_lr0.001_seed42_2layer_100dig_64d | 100 | 4 | 3.97 | 0 | 0.0% | 100 | 0.0048 | 0.9878 | 0.9287 | 0.0262 |
| sae_d100_k4_lr0.001_seed43_2layer_100dig_64d | 100 | 4 | 3.97 | 1 | 1.0% | 99 | 0.0054 | 0.9863 | 0.9287 | 0.0262 |

## Best Models by top_k


### top_k = 1

- Best reconstruction (MSE): **sae_d256_k1_2layer_100dig_64d** (MSE: 0.1303, d_sae=256)
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
| sae_d256_k1_2layer_100dig_64d | 0.0001 | 0.0186 | 0.0073 |
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
| sae_d100_k4_50ksteps_2layer_100dig_64d | 0.0199 | 0.4818 | 0.0389 |
| sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d | 0.0204 | 0.7053 | 0.0405 |
| sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d | 0.0200 | 0.5077 | 0.0410 |
| sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d | 0.0200 | 0.8354 | 0.0394 |
| sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d | 0.0201 | 0.6384 | 0.0409 |
| sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d | 0.0200 | 0.4710 | 0.0395 |
| sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d | 0.0199 | 0.6538 | 0.0406 |
| sae_d100_k4_lr0.001_seed42_2layer_100dig_64d | 0.0199 | 0.6663 | 0.0397 |
| sae_d100_k4_lr0.001_seed43_2layer_100dig_64d | 0.0199 | 0.4792 | 0.0401 |

## Special Features (Attention-Correlated)

| Model | N Special | Special % | Max Corr | Mean Abs Corr |
|-------|-----------|-----------|----------|---------------|
| sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.4767 | 0.0408 |
| sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.5601 | 0.0370 |
| sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.5165 | 0.0433 |
| sae_d50_k1_lr0.0003_seed42_2layer_100dig_64d | 0 | 0.0% | 0.4710 | 0.0430 |
| sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d | 1 | 2.0% | 0.5680 | 0.0353 |
| sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d | 1 | 2.0% | 0.5167 | 0.0430 |
| sae_d50_k1_lr0.0004_seed42_2layer_100dig_64d | 0 | 0.0% | 0.4710 | 0.0430 |
| sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d | 1 | 2.0% | 0.5690 | 0.0353 |
| sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d | 1 | 2.0% | 0.5139 | 0.0430 |
| sae_d50_k1_lr0.001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.4122 | 0.0450 |
| sae_d50_k1_lr0.001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.5584 | 0.0368 |
| sae_d50_k1_lr0.001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.5126 | 0.0436 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.2617 | 0.0462 |
| sae_d100_k1_lr0.0001_seed43_2layer_100dig_64d | 0 | 0.0% | 0.2554 | 0.0458 |
| sae_d100_k1_lr0.0001_seed44_2layer_100dig_64d | 0 | 0.0% | 0.2646 | 0.0440 |
| sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d | 0 | 0.0% | 0.2578 | 0.0454 |
| sae_d100_k1_lr0.0003_seed43_2layer_100dig_64d | 0 | 0.0% | 0.2554 | 0.0454 |
| sae_d100_k1_lr0.0003_seed44_2layer_100dig_64d | 0 | 0.0% | 0.2520 | 0.0461 |
| sae_d100_k1_lr0.001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.2113 | 0.0441 |
| sae_d100_k1_lr0.001_seed43_2layer_100dig_64d | 0 | 0.0% | 0.2553 | 0.0449 |
| sae_d256_k1_2layer_100dig_64d | 0 | 0.0% | 0.2405 | 0.0243 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 2 | 4.0% | 0.7469 | 0.0334 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8118 | 0.0216 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7543 | 0.0196 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 2 | 4.0% | 0.7539 | 0.0337 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8126 | 0.0219 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7589 | 0.0201 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 2 | 4.0% | 0.7448 | 0.0341 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8127 | 0.0219 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7537 | 0.0193 |
| sae_d50_k2_lr0.001_seed42_2layer_100dig_64d | 2 | 4.0% | 0.7486 | 0.0337 |
| sae_d50_k2_lr0.001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8126 | 0.0221 |
| sae_d50_k2_lr0.001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7629 | 0.0204 |
| sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.6033 | 0.0190 |
| sae_d100_k2_lr0.0001_seed43_2layer_100dig_64d | 0 | 0.0% | 0.4784 | 0.0215 |
| sae_d100_k2_lr0.0001_seed44_2layer_100dig_64d | 0 | 0.0% | 0.3968 | 0.0335 |
| sae_d100_k2_lr0.0003_seed42_2layer_100dig_64d | 0 | 0.0% | 0.1679 | 0.0239 |
| sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d | 1 | 1.0% | 0.5506 | 0.0226 |
| sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d | 0 | 0.0% | 0.1840 | 0.0214 |
| sae_d100_k2_lr0.001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.0970 | 0.0211 |
| sae_d100_k2_lr0.001_seed43_2layer_100dig_64d | 0 | 0.0% | 0.4436 | 0.0221 |
| sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8040 | 0.0204 |
| sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.7610 | 0.0270 |
| sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7986 | 0.0209 |
| sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8027 | 0.0205 |
| sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d | 1 | 2.0% | 0.7625 | 0.0272 |
| sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8033 | 0.0206 |
| sae_d50_k3_lr0.001_seed42_2layer_100dig_64d | 1 | 2.0% | 0.7989 | 0.0203 |
| sae_d50_k3_lr0.001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8303 | 0.0235 |
| sae_d50_k3_lr0.001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8000 | 0.0203 |
| sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8442 | 0.0106 |
| sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d | 1 | 1.0% | 0.8137 | 0.0101 |
| sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d | 1 | 1.0% | 0.8167 | 0.0101 |
| sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8375 | 0.0106 |
| sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d | 1 | 1.0% | 0.8066 | 0.0102 |
| sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d | 1 | 1.0% | 0.7996 | 0.0102 |
| sae_d100_k3_lr0.001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8350 | 0.0105 |
| sae_d100_k3_lr0.001_seed43_2layer_100dig_64d | 1 | 1.0% | 0.8122 | 0.0101 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8193 | 0.0209 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8474 | 0.0215 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7901 | 0.0244 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8242 | 0.0209 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8449 | 0.0218 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8152 | 0.0206 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8290 | 0.0207 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8449 | 0.0218 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8082 | 0.0203 |
| sae_d50_k4_lr0.001_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8201 | 0.0203 |
| sae_d50_k4_lr0.001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8445 | 0.0215 |
| sae_d50_k4_lr0.001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8043 | 0.0206 |
| sae_d100_k4_50ksteps_2layer_100dig_64d | 2 | 2.0% | 0.8035 | 0.0171 |
| sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8456 | 0.0145 |
| sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d | 2 | 2.0% | 0.7802 | 0.0173 |
| sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d | 1 | 1.0% | 0.7933 | 0.0099 |
| sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d | 2 | 2.0% | 0.8276 | 0.0154 |
| sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d | 2 | 2.0% | 0.7586 | 0.0170 |
| sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d | 1 | 1.0% | 0.8184 | 0.0150 |
| sae_d100_k4_lr0.001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8067 | 0.0141 |
| sae_d100_k4_lr0.001_seed43_2layer_100dig_64d | 2 | 2.0% | 0.7746 | 0.0173 |

### Top Special Features by Model


**sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 46: d2_favoring, corr=-0.5601

**sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.5165

**sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 46: d2_favoring, corr=-0.5680

**sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.5167

**sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d:**
- Feature 46: d2_favoring, corr=-0.5690

**sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.5139

**sae_d50_k1_lr0.001_seed43_2layer_100dig_64d:**
- Feature 46: d2_favoring, corr=-0.5584

**sae_d50_k1_lr0.001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.5126

**sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7469
- Feature 18: d1_favoring, corr=0.6844

**sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8118

**sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7543

**sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7539
- Feature 18: d1_favoring, corr=0.6860

**sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8126

**sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7589

**sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7448
- Feature 18: d1_favoring, corr=0.7008

**sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8127

**sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7537

**sae_d50_k2_lr0.001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7486
- Feature 18: d1_favoring, corr=0.6788

**sae_d50_k2_lr0.001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8126

**sae_d50_k2_lr0.001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7629

**sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.6033

**sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.5506

**sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8040

**sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.7610

**sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7986

**sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8027

**sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.7625

**sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8033

**sae_d50_k3_lr0.001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7989

**sae_d50_k3_lr0.001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8303

**sae_d50_k3_lr0.001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8000

**sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8442

**sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.8137

**sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 30: d2_favoring, corr=-0.8167

**sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8375

**sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.8066

**sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 30: d2_favoring, corr=-0.7996

**sae_d100_k3_lr0.001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8350

**sae_d100_k3_lr0.001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.8122

**sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8193

**sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8474

**sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 40: d1_favoring, corr=0.7901

**sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8242

**sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8449

**sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8152

**sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8290

**sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8449

**sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8082

**sae_d50_k4_lr0.001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8201

**sae_d50_k4_lr0.001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8445

**sae_d50_k4_lr0.001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8043

**sae_d100_k4_50ksteps_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8035
- Feature 84: d2_favoring, corr=-0.7142

**sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8456

**sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.7802
- Feature 68: d1_favoring, corr=0.7504

**sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 30: d2_favoring, corr=-0.7933

**sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8276
- Feature 72: d2_favoring, corr=-0.5172

**sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 68: d1_favoring, corr=0.7586
- Feature 47: d2_favoring, corr=-0.7569

**sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 30: d2_favoring, corr=-0.8184

**sae_d100_k4_lr0.001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8067

**sae_d100_k4_lr0.001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.7746
- Feature 68: d1_favoring, corr=0.7593

## Analysis

- **L0**: Average number of active features per sample (lower = sparser)
- **Dead %**: Percentage of features that never fire (lower = better utilization)
- **MSE**: Mean squared reconstruction error (lower = better reconstruction)
- **Exp Var**: Explained variance (higher = better reconstruction)
- **Special Features**: Features with |correlation| > 0.5 with attention difference (alpha_d1 - alpha_d2)
