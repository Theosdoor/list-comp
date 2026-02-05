# SAE Sweep Comparison Report

Compared 79 SAE models from sweep runs on 2000 samples.


## Summary Table

| Model | d_sae | k | L0 | Dead | Dead % | Alive | MSE | Exp Var | Recon Acc | Acc Drop |
|-------|-------|---|----|----|--------|-------|-----|---------|-----------|----------|
| sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d | 50 | 1 | 0.97 | 0 | 0.0% | 50 | 0.1791 | 0.5256 | 0.3003 | 0.6142 |
| sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d | 50 | 1 | 0.97 | 0 | 0.0% | 50 | 0.1864 | 0.5064 | 0.3083 | 0.6062 |
| sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d | 50 | 1 | 0.98 | 0 | 0.0% | 50 | 0.1789 | 0.5261 | 0.2995 | 0.6150 |
| sae_d50_k1_lr0.0003_seed42_2layer_100dig_64d | 50 | 1 | 0.97 | 0 | 0.0% | 50 | 0.1790 | 0.5260 | 0.2943 | 0.6202 |
| sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d | 50 | 1 | 0.96 | 0 | 0.0% | 50 | 0.1880 | 0.5021 | 0.2988 | 0.6158 |
| sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d | 50 | 1 | 0.98 | 0 | 0.0% | 50 | 0.1792 | 0.5253 | 0.2988 | 0.6158 |
| sae_d50_k1_lr0.0004_seed42_2layer_100dig_64d | 50 | 1 | 0.97 | 0 | 0.0% | 50 | 0.1789 | 0.5260 | 0.2943 | 0.6202 |
| sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d | 50 | 1 | 0.96 | 0 | 0.0% | 50 | 0.1878 | 0.5025 | 0.3003 | 0.6142 |
| sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d | 50 | 1 | 0.97 | 0 | 0.0% | 50 | 0.1811 | 0.5204 | 0.2973 | 0.6172 |
| sae_d50_k1_lr0.001_seed42_2layer_100dig_64d | 50 | 1 | 0.96 | 0 | 0.0% | 50 | 0.1795 | 0.5245 | 0.2973 | 0.6172 |
| sae_d50_k1_lr0.001_seed43_2layer_100dig_64d | 50 | 1 | 0.96 | 0 | 0.0% | 50 | 0.1876 | 0.5030 | 0.2988 | 0.6158 |
| sae_d50_k1_lr0.001_seed44_2layer_100dig_64d | 50 | 1 | 0.97 | 0 | 0.0% | 50 | 0.1814 | 0.5194 | 0.2980 | 0.6165 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 100 | 1 | 0.93 | 1 | 1.0% | 99 | 0.1510 | 0.6001 | 0.3295 | 0.5850 |
| sae_d100_k1_lr0.0001_seed43_2layer_100dig_64d | 100 | 1 | 0.93 | 1 | 1.0% | 99 | 0.1518 | 0.5980 | 0.3260 | 0.5885 |
| sae_d100_k1_lr0.0001_seed44_2layer_100dig_64d | 100 | 1 | 0.92 | 2 | 2.0% | 98 | 0.1512 | 0.5996 | 0.3360 | 0.5785 |
| sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d | 100 | 1 | 0.93 | 2 | 2.0% | 98 | 0.1497 | 0.6035 | 0.3360 | 0.5785 |
| sae_d100_k1_lr0.0003_seed43_2layer_100dig_64d | 100 | 1 | 0.93 | 1 | 1.0% | 99 | 0.1530 | 0.5948 | 0.3300 | 0.5845 |
| sae_d100_k1_lr0.0003_seed44_2layer_100dig_64d | 100 | 1 | 0.93 | 0 | 0.0% | 100 | 0.1508 | 0.6004 | 0.3375 | 0.5770 |
| sae_d100_k1_lr0.001_seed42_2layer_100dig_64d | 100 | 1 | 0.90 | 2 | 2.0% | 98 | 0.1551 | 0.5893 | 0.3282 | 0.5862 |
| sae_d100_k1_lr0.001_seed43_2layer_100dig_64d | 100 | 1 | 0.93 | 0 | 0.0% | 100 | 0.1521 | 0.5972 | 0.3327 | 0.5817 |
| sae_d256_k1_2layer_100dig_64d | 256 | 1 | 0.94 | 121 | 47.3% | 135 | 0.1513 | 0.5993 | 0.3498 | 0.5647 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1418 | 0.6243 | 0.4310 | 0.4835 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1441 | 0.6184 | 0.4447 | 0.4698 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1397 | 0.6299 | 0.4460 | 0.4685 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1421 | 0.6236 | 0.4245 | 0.4900 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1437 | 0.6195 | 0.4335 | 0.4810 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1394 | 0.6309 | 0.4480 | 0.4665 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1409 | 0.6269 | 0.4335 | 0.4810 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1437 | 0.6195 | 0.4338 | 0.4807 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 50 | 2 | 2.00 | 0 | 0.0% | 50 | 0.1389 | 0.6320 | 0.4547 | 0.4597 |
| sae_d50_k2_lr0.001_seed42_2layer_100dig_64d | 50 | 2 | 1.99 | 0 | 0.0% | 50 | 0.1413 | 0.6258 | 0.4358 | 0.4787 |
| sae_d50_k2_lr0.001_seed43_2layer_100dig_64d | 50 | 2 | 2.03 | 0 | 0.0% | 50 | 0.1422 | 0.6234 | 0.4377 | 0.4768 |
| sae_d50_k2_lr0.001_seed44_2layer_100dig_64d | 50 | 2 | 2.01 | 0 | 0.0% | 50 | 0.1385 | 0.6332 | 0.4385 | 0.4760 |
| sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d | 100 | 2 | 1.97 | 1 | 1.0% | 99 | 0.0421 | 0.8885 | 0.6078 | 0.3067 |
| sae_d100_k2_lr0.0001_seed43_2layer_100dig_64d | 100 | 2 | 1.94 | 1 | 1.0% | 99 | 0.0436 | 0.8846 | 0.5992 | 0.3153 |
| sae_d100_k2_lr0.0001_seed44_2layer_100dig_64d | 100 | 2 | 2.26 | 1 | 1.0% | 99 | 0.0671 | 0.8222 | 0.6390 | 0.2755 |
| sae_d100_k2_lr0.0003_seed42_2layer_100dig_64d | 100 | 2 | 2.05 | 2 | 2.0% | 98 | 0.0530 | 0.8596 | 0.6270 | 0.2875 |
| sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d | 100 | 2 | 1.96 | 1 | 1.0% | 99 | 0.0433 | 0.8854 | 0.6162 | 0.2983 |
| sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d | 100 | 2 | 1.97 | 0 | 0.0% | 100 | 0.0412 | 0.8910 | 0.6538 | 0.2607 |
| sae_d100_k2_lr0.001_seed42_2layer_100dig_64d | 100 | 2 | 2.03 | 1 | 1.0% | 99 | 0.0524 | 0.8611 | 0.6565 | 0.2580 |
| sae_d100_k2_lr0.001_seed43_2layer_100dig_64d | 100 | 2 | 1.99 | 1 | 1.0% | 99 | 0.0512 | 0.8643 | 0.6352 | 0.2792 |
| sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d | 50 | 3 | 3.01 | 0 | 0.0% | 50 | 0.1240 | 0.6717 | 0.5212 | 0.3932 |
| sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d | 50 | 3 | 3.06 | 0 | 0.0% | 50 | 0.1259 | 0.6666 | 0.5155 | 0.3990 |
| sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d | 50 | 3 | 3.03 | 0 | 0.0% | 50 | 0.1229 | 0.6746 | 0.5282 | 0.3862 |
| sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d | 50 | 3 | 3.05 | 0 | 0.0% | 50 | 0.1249 | 0.6693 | 0.5228 | 0.3917 |
| sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d | 50 | 3 | 3.07 | 0 | 0.0% | 50 | 0.1260 | 0.6661 | 0.5218 | 0.3927 |
| sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d | 50 | 3 | 3.02 | 0 | 0.0% | 50 | 0.1219 | 0.6772 | 0.5235 | 0.3910 |
| sae_d50_k3_lr0.001_seed42_2layer_100dig_64d | 50 | 3 | 3.02 | 0 | 0.0% | 50 | 0.1240 | 0.6715 | 0.5170 | 0.3975 |
| sae_d50_k3_lr0.001_seed43_2layer_100dig_64d | 50 | 3 | 3.03 | 0 | 0.0% | 50 | 0.1243 | 0.6707 | 0.5298 | 0.3847 |
| sae_d50_k3_lr0.001_seed44_2layer_100dig_64d | 50 | 3 | 3.04 | 0 | 0.0% | 50 | 0.1210 | 0.6794 | 0.5232 | 0.3912 |
| sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d | 100 | 3 | 2.98 | 1 | 1.0% | 99 | 0.0103 | 0.9728 | 0.8680 | 0.0465 |
| sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d | 100 | 3 | 2.89 | 0 | 0.0% | 100 | 0.0079 | 0.9790 | 0.8752 | 0.0393 |
| sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d | 100 | 3 | 2.87 | 0 | 0.0% | 100 | 0.0087 | 0.9770 | 0.8655 | 0.0490 |
| sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d | 100 | 3 | 3.11 | 0 | 0.0% | 100 | 0.0097 | 0.9744 | 0.8678 | 0.0467 |
| sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d | 100 | 3 | 2.95 | 0 | 0.0% | 100 | 0.0083 | 0.9781 | 0.8738 | 0.0407 |
| sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d | 100 | 3 | 3.08 | 0 | 0.0% | 100 | 0.0089 | 0.9763 | 0.8688 | 0.0457 |
| sae_d100_k3_lr0.001_seed42_2layer_100dig_64d | 100 | 3 | 3.02 | 1 | 1.0% | 99 | 0.0103 | 0.9726 | 0.8740 | 0.0405 |
| sae_d100_k3_lr0.001_seed43_2layer_100dig_64d | 100 | 3 | 3.01 | 0 | 0.0% | 100 | 0.0082 | 0.9783 | 0.8740 | 0.0405 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 50 | 4 | 4.03 | 0 | 0.0% | 50 | 0.1108 | 0.7066 | 0.5920 | 0.3225 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 50 | 4 | 4.14 | 0 | 0.0% | 50 | 0.1092 | 0.7107 | 0.6198 | 0.2947 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 50 | 4 | 4.01 | 0 | 0.0% | 50 | 0.1124 | 0.7023 | 0.6232 | 0.2913 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 50 | 4 | 4.09 | 0 | 0.0% | 50 | 0.1108 | 0.7066 | 0.5965 | 0.3180 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 50 | 4 | 4.06 | 0 | 0.0% | 50 | 0.1117 | 0.7041 | 0.6365 | 0.2780 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 50 | 4 | 4.03 | 0 | 0.0% | 50 | 0.1096 | 0.7096 | 0.6070 | 0.3075 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 50 | 4 | 4.06 | 0 | 0.0% | 50 | 0.1106 | 0.7070 | 0.5857 | 0.3287 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 50 | 4 | 4.07 | 0 | 0.0% | 50 | 0.1117 | 0.7040 | 0.6370 | 0.2775 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 50 | 4 | 4.06 | 0 | 0.0% | 50 | 0.1103 | 0.7080 | 0.6118 | 0.3027 |
| sae_d50_k4_lr0.001_seed42_2layer_100dig_64d | 50 | 4 | 4.07 | 0 | 0.0% | 50 | 0.1106 | 0.7071 | 0.6110 | 0.3035 |
| sae_d50_k4_lr0.001_seed43_2layer_100dig_64d | 50 | 4 | 4.06 | 0 | 0.0% | 50 | 0.1099 | 0.7089 | 0.6232 | 0.2913 |
| sae_d50_k4_lr0.001_seed44_2layer_100dig_64d | 50 | 4 | 4.02 | 0 | 0.0% | 50 | 0.1094 | 0.7103 | 0.6105 | 0.3040 |
| sae_d100_k4_50ksteps_2layer_100dig_64d | 100 | 4 | 3.94 | 0 | 0.0% | 100 | 0.0058 | 0.9846 | 0.8885 | 0.0260 |
| sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d | 100 | 4 | 4.22 | 0 | 0.0% | 100 | 0.0088 | 0.9767 | 0.8758 | 0.0387 |
| sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d | 100 | 4 | 4.26 | 0 | 0.0% | 100 | 0.0093 | 0.9753 | 0.8718 | 0.0427 |
| sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d | 100 | 4 | 4.09 | 0 | 0.0% | 100 | 0.0086 | 0.9773 | 0.8700 | 0.0445 |
| sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d | 100 | 4 | 4.22 | 0 | 0.0% | 100 | 0.0093 | 0.9753 | 0.8692 | 0.0453 |
| sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d | 100 | 4 | 4.11 | 0 | 0.0% | 100 | 0.0097 | 0.9743 | 0.8670 | 0.0475 |
| sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d | 100 | 4 | 4.19 | 0 | 0.0% | 100 | 0.0093 | 0.9754 | 0.8728 | 0.0417 |
| sae_d100_k4_lr0.001_seed42_2layer_100dig_64d | 100 | 4 | 4.14 | 0 | 0.0% | 100 | 0.0093 | 0.9753 | 0.8705 | 0.0440 |
| sae_d100_k4_lr0.001_seed43_2layer_100dig_64d | 100 | 4 | 4.08 | 1 | 1.0% | 99 | 0.0098 | 0.9742 | 0.8710 | 0.0435 |

## Best Models by top_k


### top_k = 1

- Best reconstruction (MSE): **sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d** (MSE: 0.1497, d_sae=100)
- Fewest dead features: **sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

### top_k = 2

- Best reconstruction (MSE): **sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d** (MSE: 0.0412, d_sae=100)
- Fewest dead features: **sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

### top_k = 3

- Best reconstruction (MSE): **sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d** (MSE: 0.0079, d_sae=100)
- Fewest dead features: **sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

### top_k = 4

- Best reconstruction (MSE): **sae_d100_k4_50ksteps_2layer_100dig_64d** (MSE: 0.0058, d_sae=100)
- Fewest dead features: **sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d** (0.0%, d_sae=50)

## Firing Rate Statistics

| Model | Min Firing | Max Firing | Mean Firing |
|-------|------------|------------|-------------|
| sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d | 0.0110 | 0.0900 | 0.0194 |
| sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d | 0.0105 | 0.1085 | 0.0195 |
| sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d | 0.0105 | 0.1080 | 0.0195 |
| sae_d50_k1_lr0.0003_seed42_2layer_100dig_64d | 0.0110 | 0.0885 | 0.0194 |
| sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d | 0.0105 | 0.1140 | 0.0193 |
| sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d | 0.0105 | 0.1065 | 0.0195 |
| sae_d50_k1_lr0.0004_seed42_2layer_100dig_64d | 0.0110 | 0.0885 | 0.0194 |
| sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d | 0.0105 | 0.1145 | 0.0193 |
| sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d | 0.0095 | 0.1025 | 0.0194 |
| sae_d50_k1_lr0.001_seed42_2layer_100dig_64d | 0.0110 | 0.0615 | 0.0193 |
| sae_d50_k1_lr0.001_seed43_2layer_100dig_64d | 0.0105 | 0.1070 | 0.0192 |
| sae_d50_k1_lr0.001_seed44_2layer_100dig_64d | 0.0095 | 0.1025 | 0.0193 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 0.0005 | 0.0185 | 0.0094 |
| sae_d100_k1_lr0.0001_seed43_2layer_100dig_64d | 0.0010 | 0.0245 | 0.0094 |
| sae_d100_k1_lr0.0001_seed44_2layer_100dig_64d | 0.0015 | 0.0175 | 0.0094 |
| sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d | 0.0010 | 0.0180 | 0.0094 |
| sae_d100_k1_lr0.0003_seed43_2layer_100dig_64d | 0.0020 | 0.0245 | 0.0094 |
| sae_d100_k1_lr0.0003_seed44_2layer_100dig_64d | 0.0015 | 0.0170 | 0.0093 |
| sae_d100_k1_lr0.001_seed42_2layer_100dig_64d | 0.0015 | 0.0200 | 0.0092 |
| sae_d100_k1_lr0.001_seed43_2layer_100dig_64d | 0.0015 | 0.0235 | 0.0093 |
| sae_d256_k1_2layer_100dig_64d | 0.0005 | 0.0170 | 0.0069 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 0.0155 | 0.3405 | 0.0399 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 0.0155 | 0.4365 | 0.0397 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 0.0145 | 0.4455 | 0.0399 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 0.0160 | 0.3565 | 0.0398 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 0.0155 | 0.4355 | 0.0401 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 0.0145 | 0.4420 | 0.0399 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 0.0155 | 0.3315 | 0.0400 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 0.0155 | 0.4355 | 0.0400 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 0.0145 | 0.4510 | 0.0400 |
| sae_d50_k2_lr0.001_seed42_2layer_100dig_64d | 0.0160 | 0.3390 | 0.0397 |
| sae_d50_k2_lr0.001_seed43_2layer_100dig_64d | 0.0160 | 0.4385 | 0.0405 |
| sae_d50_k2_lr0.001_seed44_2layer_100dig_64d | 0.0145 | 0.4250 | 0.0401 |
| sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d | 0.0015 | 0.1760 | 0.0199 |
| sae_d100_k2_lr0.0001_seed43_2layer_100dig_64d | 0.0060 | 0.0965 | 0.0196 |
| sae_d100_k2_lr0.0001_seed44_2layer_100dig_64d | 0.0125 | 0.0670 | 0.0228 |
| sae_d100_k2_lr0.0003_seed42_2layer_100dig_64d | 0.0070 | 0.0680 | 0.0209 |
| sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d | 0.0085 | 0.1140 | 0.0198 |
| sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d | 0.0070 | 0.0430 | 0.0197 |
| sae_d100_k2_lr0.001_seed42_2layer_100dig_64d | 0.0130 | 0.0640 | 0.0205 |
| sae_d100_k2_lr0.001_seed43_2layer_100dig_64d | 0.0075 | 0.0665 | 0.0201 |
| sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d | 0.0165 | 0.4955 | 0.0603 |
| sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d | 0.0185 | 0.4265 | 0.0612 |
| sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d | 0.0145 | 0.5020 | 0.0606 |
| sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d | 0.0175 | 0.5060 | 0.0609 |
| sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d | 0.0180 | 0.4240 | 0.0614 |
| sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d | 0.0170 | 0.4930 | 0.0604 |
| sae_d50_k3_lr0.001_seed42_2layer_100dig_64d | 0.0155 | 0.5210 | 0.0604 |
| sae_d50_k3_lr0.001_seed43_2layer_100dig_64d | 0.0185 | 0.4865 | 0.0607 |
| sae_d50_k3_lr0.001_seed44_2layer_100dig_64d | 0.0170 | 0.5025 | 0.0607 |
| sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d | 0.0140 | 0.7475 | 0.0301 |
| sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d | 0.0140 | 0.7225 | 0.0289 |
| sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d | 0.0140 | 0.7200 | 0.0287 |
| sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d | 0.0140 | 0.7365 | 0.0311 |
| sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d | 0.0140 | 0.7045 | 0.0295 |
| sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d | 0.0140 | 0.7165 | 0.0308 |
| sae_d100_k3_lr0.001_seed42_2layer_100dig_64d | 0.0140 | 0.7175 | 0.0305 |
| sae_d100_k3_lr0.001_seed43_2layer_100dig_64d | 0.0160 | 0.7335 | 0.0301 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 0.0160 | 0.5615 | 0.0805 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 0.0195 | 0.5665 | 0.0827 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 0.0185 | 0.5100 | 0.0803 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 0.0170 | 0.5550 | 0.0819 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 0.0220 | 0.5525 | 0.0813 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 0.0220 | 0.5760 | 0.0807 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 0.0160 | 0.5520 | 0.0812 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 0.0210 | 0.5530 | 0.0813 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 0.0180 | 0.5890 | 0.0811 |
| sae_d50_k4_lr0.001_seed42_2layer_100dig_64d | 0.0175 | 0.5700 | 0.0814 |
| sae_d50_k4_lr0.001_seed43_2layer_100dig_64d | 0.0225 | 0.5650 | 0.0813 |
| sae_d50_k4_lr0.001_seed44_2layer_100dig_64d | 0.0180 | 0.5585 | 0.0805 |
| sae_d100_k4_50ksteps_2layer_100dig_64d | 0.0175 | 0.4820 | 0.0394 |
| sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d | 0.0170 | 0.7180 | 0.0422 |
| sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d | 0.0185 | 0.5035 | 0.0426 |
| sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d | 0.0150 | 0.8470 | 0.0409 |
| sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d | 0.0180 | 0.6440 | 0.0422 |
| sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d | 0.0145 | 0.4645 | 0.0412 |
| sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d | 0.0165 | 0.6585 | 0.0419 |
| sae_d100_k4_lr0.001_seed42_2layer_100dig_64d | 0.0155 | 0.6685 | 0.0414 |
| sae_d100_k4_lr0.001_seed43_2layer_100dig_64d | 0.0140 | 0.4760 | 0.0412 |

## Special Features (Attention-Correlated)

| Model | N Special | Special % | Max Corr | Mean Abs Corr |
|-------|-----------|-----------|----------|---------------|
| sae_d50_k1_lr0.0001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.4609 | 0.0458 |
| sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.6042 | 0.0447 |
| sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.5584 | 0.0464 |
| sae_d50_k1_lr0.0003_seed42_2layer_100dig_64d | 0 | 0.0% | 0.4574 | 0.0473 |
| sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d | 1 | 2.0% | 0.6146 | 0.0433 |
| sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d | 1 | 2.0% | 0.5602 | 0.0463 |
| sae_d50_k1_lr0.0004_seed42_2layer_100dig_64d | 0 | 0.0% | 0.4574 | 0.0473 |
| sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d | 1 | 2.0% | 0.6159 | 0.0433 |
| sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d | 1 | 2.0% | 0.5542 | 0.0467 |
| sae_d50_k1_lr0.001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.4027 | 0.0501 |
| sae_d50_k1_lr0.001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.6023 | 0.0443 |
| sae_d50_k1_lr0.001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.5500 | 0.0467 |
| sae_d100_k1_lr0.0001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.3368 | 0.0477 |
| sae_d100_k1_lr0.0001_seed43_2layer_100dig_64d | 0 | 0.0% | 0.2555 | 0.0479 |
| sae_d100_k1_lr0.0001_seed44_2layer_100dig_64d | 0 | 0.0% | 0.3321 | 0.0464 |
| sae_d100_k1_lr0.0003_seed42_2layer_100dig_64d | 0 | 0.0% | 0.3143 | 0.0477 |
| sae_d100_k1_lr0.0003_seed43_2layer_100dig_64d | 0 | 0.0% | 0.2555 | 0.0476 |
| sae_d100_k1_lr0.0003_seed44_2layer_100dig_64d | 0 | 0.0% | 0.3141 | 0.0490 |
| sae_d100_k1_lr0.001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.2374 | 0.0442 |
| sae_d100_k1_lr0.001_seed43_2layer_100dig_64d | 0 | 0.0% | 0.2615 | 0.0465 |
| sae_d256_k1_2layer_100dig_64d | 0 | 0.0% | 0.2256 | 0.0255 |
| sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d | 2 | 4.0% | 0.7663 | 0.0433 |
| sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.7959 | 0.0356 |
| sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7794 | 0.0325 |
| sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d | 2 | 4.0% | 0.7735 | 0.0436 |
| sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d | 1 | 2.0% | 0.7944 | 0.0356 |
| sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7817 | 0.0327 |
| sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d | 2 | 4.0% | 0.7642 | 0.0434 |
| sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d | 1 | 2.0% | 0.7945 | 0.0355 |
| sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7780 | 0.0317 |
| sae_d50_k2_lr0.001_seed42_2layer_100dig_64d | 2 | 4.0% | 0.7676 | 0.0431 |
| sae_d50_k2_lr0.001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.7946 | 0.0358 |
| sae_d50_k2_lr0.001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7848 | 0.0309 |
| sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.5825 | 0.0299 |
| sae_d100_k2_lr0.0001_seed43_2layer_100dig_64d | 1 | 1.0% | 0.5110 | 0.0297 |
| sae_d100_k2_lr0.0001_seed44_2layer_100dig_64d | 0 | 0.0% | 0.4341 | 0.0392 |
| sae_d100_k2_lr0.0003_seed42_2layer_100dig_64d | 0 | 0.0% | 0.1797 | 0.0304 |
| sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d | 1 | 1.0% | 0.5675 | 0.0324 |
| sae_d100_k2_lr0.0003_seed44_2layer_100dig_64d | 0 | 0.0% | 0.1764 | 0.0268 |
| sae_d100_k2_lr0.001_seed42_2layer_100dig_64d | 0 | 0.0% | 0.1219 | 0.0265 |
| sae_d100_k2_lr0.001_seed43_2layer_100dig_64d | 0 | 0.0% | 0.4904 | 0.0295 |
| sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8178 | 0.0314 |
| sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.7420 | 0.0399 |
| sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8162 | 0.0312 |
| sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8173 | 0.0317 |
| sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d | 1 | 2.0% | 0.7438 | 0.0396 |
| sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8192 | 0.0311 |
| sae_d50_k3_lr0.001_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8156 | 0.0293 |
| sae_d50_k3_lr0.001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8139 | 0.0355 |
| sae_d50_k3_lr0.001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8160 | 0.0303 |
| sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8342 | 0.0254 |
| sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d | 1 | 1.0% | 0.8297 | 0.0253 |
| sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d | 1 | 1.0% | 0.8337 | 0.0250 |
| sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8275 | 0.0246 |
| sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d | 1 | 1.0% | 0.8250 | 0.0250 |
| sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d | 1 | 1.0% | 0.8191 | 0.0248 |
| sae_d100_k3_lr0.001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8271 | 0.0248 |
| sae_d100_k3_lr0.001_seed43_2layer_100dig_64d | 1 | 1.0% | 0.8285 | 0.0250 |
| sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8318 | 0.0302 |
| sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8310 | 0.0363 |
| sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.7709 | 0.0366 |
| sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8374 | 0.0282 |
| sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8276 | 0.0345 |
| sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8285 | 0.0315 |
| sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8392 | 0.0304 |
| sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8277 | 0.0347 |
| sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8241 | 0.0331 |
| sae_d50_k4_lr0.001_seed42_2layer_100dig_64d | 1 | 2.0% | 0.8342 | 0.0304 |
| sae_d50_k4_lr0.001_seed43_2layer_100dig_64d | 1 | 2.0% | 0.8309 | 0.0344 |
| sae_d50_k4_lr0.001_seed44_2layer_100dig_64d | 1 | 2.0% | 0.8207 | 0.0326 |
| sae_d100_k4_50ksteps_2layer_100dig_64d | 2 | 2.0% | 0.7899 | 0.0323 |
| sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.8333 | 0.0290 |
| sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d | 2 | 2.0% | 0.7943 | 0.0316 |
| sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d | 1 | 1.0% | 0.8108 | 0.0244 |
| sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d | 2 | 2.0% | 0.8185 | 0.0301 |
| sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d | 2 | 2.0% | 0.7690 | 0.0313 |
| sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d | 1 | 1.0% | 0.8345 | 0.0305 |
| sae_d100_k4_lr0.001_seed42_2layer_100dig_64d | 1 | 1.0% | 0.7946 | 0.0296 |
| sae_d100_k4_lr0.001_seed43_2layer_100dig_64d | 2 | 2.0% | 0.7853 | 0.0324 |

### Top Special Features by Model


**sae_d50_k1_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 46: d2_favoring, corr=-0.6042

**sae_d50_k1_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.5584

**sae_d50_k1_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 46: d2_favoring, corr=-0.6146

**sae_d50_k1_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.5602

**sae_d50_k1_lr0.0004_seed43_2layer_100dig_64d:**
- Feature 46: d2_favoring, corr=-0.6159

**sae_d50_k1_lr0.0004_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.5542

**sae_d50_k1_lr0.001_seed43_2layer_100dig_64d:**
- Feature 46: d2_favoring, corr=-0.6023

**sae_d50_k1_lr0.001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.5500

**sae_d50_k2_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7663
- Feature 18: d1_favoring, corr=0.6628

**sae_d50_k2_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.7959

**sae_d50_k2_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7794

**sae_d50_k2_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7735
- Feature 18: d1_favoring, corr=0.6663

**sae_d50_k2_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.7944

**sae_d50_k2_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7817

**sae_d50_k2_lr0.0004_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7642
- Feature 18: d1_favoring, corr=0.6837

**sae_d50_k2_lr0.0004_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.7945

**sae_d50_k2_lr0.0004_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7780

**sae_d50_k2_lr0.001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.7676
- Feature 18: d1_favoring, corr=0.6592

**sae_d50_k2_lr0.001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.7946

**sae_d50_k2_lr0.001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.7848

**sae_d100_k2_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.5825

**sae_d100_k2_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.5110

**sae_d100_k2_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.5675

**sae_d50_k3_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8178

**sae_d50_k3_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.7420

**sae_d50_k3_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8162

**sae_d50_k3_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8173

**sae_d50_k3_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.7438

**sae_d50_k3_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8192

**sae_d50_k3_lr0.001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8156

**sae_d50_k3_lr0.001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8139

**sae_d50_k3_lr0.001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8160

**sae_d100_k3_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8342

**sae_d100_k3_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.8297

**sae_d100_k3_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 30: d2_favoring, corr=-0.8337

**sae_d100_k3_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8275

**sae_d100_k3_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.8250

**sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 30: d2_favoring, corr=-0.8191

**sae_d100_k3_lr0.001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8271

**sae_d100_k3_lr0.001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.8285

**sae_d50_k4_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8318

**sae_d50_k4_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8310

**sae_d50_k4_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 40: d1_favoring, corr=0.7709

**sae_d50_k4_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8374

**sae_d50_k4_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8276

**sae_d50_k4_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8285

**sae_d50_k4_lr0.0004_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8392

**sae_d50_k4_lr0.0004_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8277

**sae_d50_k4_lr0.0004_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8241

**sae_d50_k4_lr0.001_seed42_2layer_100dig_64d:**
- Feature 11: d2_favoring, corr=-0.8342

**sae_d50_k4_lr0.001_seed43_2layer_100dig_64d:**
- Feature 0: d1_favoring, corr=0.8309

**sae_d50_k4_lr0.001_seed44_2layer_100dig_64d:**
- Feature 43: d2_favoring, corr=-0.8207

**sae_d100_k4_50ksteps_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.7899
- Feature 84: d2_favoring, corr=-0.7318

**sae_d100_k4_lr0.0001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8333

**sae_d100_k4_lr0.0001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.7943
- Feature 68: d1_favoring, corr=0.7389

**sae_d100_k4_lr0.0001_seed44_2layer_100dig_64d:**
- Feature 30: d2_favoring, corr=-0.8108

**sae_d100_k4_lr0.0003_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.8185
- Feature 72: d2_favoring, corr=-0.5541

**sae_d100_k4_lr0.0003_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.7690
- Feature 68: d1_favoring, corr=0.7493

**sae_d100_k4_lr0.0003_seed44_2layer_100dig_64d:**
- Feature 30: d2_favoring, corr=-0.8345

**sae_d100_k4_lr0.001_seed42_2layer_100dig_64d:**
- Feature 10: d1_favoring, corr=0.7946

**sae_d100_k4_lr0.001_seed43_2layer_100dig_64d:**
- Feature 47: d2_favoring, corr=-0.7853
- Feature 68: d1_favoring, corr=0.7503

## Analysis

- **L0**: Average number of active features per sample (lower = sparser)
- **Dead %**: Percentage of features that never fire (lower = better utilization)
- **MSE**: Mean squared reconstruction error (lower = better reconstruction)
- **Exp Var**: Explained variance (higher = better reconstruction)
- **Special Features**: Features with |correlation| > 0.5 with attention difference (alpha_d1 - alpha_d2)
