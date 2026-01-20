# SAE Comparison Report

Compared 12 SAE models on 10000 samples.


## Summary Table

| Model | d_sae | k | L0 | Dead | Dead % | Alive | MSE |
|-------|-------|---|----|----|--------|-------|-----|
| old_sae | 256 | 4 | 4.00 | 92 | 35.9% | 164 | 0.0027 |
| sae_d100_k4_2layer_100dig_64d | 100 | 4 | 3.62 | 1 | 1.0% | 99 | 0.0071 |
| sae_d100_k4_50000steps_2layer_100dig_64d | 100 | 4 | 3.89 | 0 | 0.0% | 100 | 0.0045 |
| sae_d150_k2_2layer_100dig_64d | 150 | 2 | 2.08 | 46 | 30.7% | 104 | 0.0295 |
| sae_d150_k4_2layer_100dig_64d | 150 | 4 | 3.52 | 28 | 18.7% | 122 | 0.0028 |
| sae_d150_k4_50000steps_2layer_100dig_64d | 150 | 4 | 3.76 | 0 | 0.0% | 150 | 0.0027 |
| sae_d150_k8_2layer_100dig_64d | 150 | 8 | 7.70 | 0 | 0.0% | 150 | 0.0027 |
| sae_d200_50000steps_k4_2layer_100dig_64d | 200 | 4 | 3.89 | 6 | 3.0% | 194 | 0.0029 |
| sae_d200_k4_2layer_100dig_64d | 200 | 4 | 3.55 | 60 | 30.0% | 140 | 0.0027 |
| sae_d256_50000steps_k4_2layer_100dig_64d | 256 | 4 | 3.91 | 7 | 2.7% | 249 | 0.0032 |
| sae_d256_k4 | 256 | 4 | 3.63 | 109 | 42.6% | 147 | 0.0026 |
| sae_d256_k4_2layer_100dig_64d | 256 | 4 | 3.63 | 109 | 42.6% | 147 | 0.0026 |

## Firing Rate Statistics

| Model | Min Firing | Max Firing | Mean Firing |
|-------|------------|------------|-------------|
| old_sae | 0.0001 | 0.6090 | 0.0244 |
| sae_d100_k4_2layer_100dig_64d | 0.0001 | 0.4681 | 0.0365 |
| sae_d100_k4_50000steps_2layer_100dig_64d | 0.0199 | 0.4818 | 0.0389 |
| sae_d150_k2_2layer_100dig_64d | 0.0038 | 0.0739 | 0.0200 |
| sae_d150_k4_2layer_100dig_64d | 0.0031 | 0.5502 | 0.0289 |
| sae_d150_k4_50000steps_2layer_100dig_64d | 0.0020 | 0.5509 | 0.0250 |
| sae_d150_k8_2layer_100dig_64d | 0.0047 | 0.5599 | 0.0513 |
| sae_d200_50000steps_k4_2layer_100dig_64d | 0.0008 | 0.4567 | 0.0201 |
| sae_d200_k4_2layer_100dig_64d | 0.0001 | 0.4576 | 0.0254 |
| sae_d256_50000steps_k4_2layer_100dig_64d | 0.0007 | 0.6014 | 0.0157 |
| sae_d256_k4 | 0.0001 | 0.6050 | 0.0247 |
| sae_d256_k4_2layer_100dig_64d | 0.0001 | 0.6050 | 0.0247 |

## Top Features by Firing Rate

### old_sae

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 189 | 0.6090 | D1 | 75 |
| 110 | 0.1703 | D1 | 77 |
| 31 | 0.0314 | D1 | 42 |
| 132 | 0.0298 | D1 | 51 |
| 125 | 0.0287 | D1 | 52 |

### sae_d100_k4_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 10 | 0.4681 | D1 | 10 |
| 84 | 0.3596 | D1 | 70 |
| 63 | 0.0663 | D1 | 31 |
| 48 | 0.0616 | D1 | 31 |
| 19 | 0.0600 | D1 | 76 |

### sae_d100_k4_50000steps_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 10 | 0.4818 | D1 | 10 |
| 84 | 0.3709 | D1 | 77 |
| 25 | 0.0707 | D1 | 53 |
| 19 | 0.0616 | D1 | 76 |
| 36 | 0.0596 | D1 | 86 |

### sae_d150_k2_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 10 | 0.0739 | D1 | 51 |
| 6 | 0.0455 | D2 | 70 |
| 74 | 0.0310 | D1 | 29 |
| 25 | 0.0304 | D1 | 10 |
| 49 | 0.0273 | D1 | 99 |

### sae_d150_k4_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 22 | 0.5502 | D1 | 7 |
| 37 | 0.3595 | D1 | 70 |
| 93 | 0.0663 | D1 | 54 |
| 16 | 0.0489 | D1 | 22 |
| 25 | 0.0476 | D1 | 10 |

### sae_d150_k4_50000steps_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 22 | 0.5509 | D1 | 7 |
| 37 | 0.3595 | D1 | 70 |
| 93 | 0.0675 | D1 | 54 |
| 49 | 0.0416 | D1 | 99 |
| 60 | 0.0405 | D1 | 95 |

### sae_d150_k8_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 22 | 0.5599 | D1 | 10 |
| 37 | 0.4110 | D1 | 70 |
| 16 | 0.1576 | D1 | 54 |
| 52 | 0.1108 | D1 | 57 |
| 60 | 0.1081 | D1 | 29 |

### sae_d200_50000steps_k4_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 121 | 0.4567 | D1 | 24 |
| 72 | 0.4298 | D1 | 51 |
| 61 | 0.0405 | D1 | 40 |
| 171 | 0.0304 | D1 | 78 |
| 102 | 0.0298 | D1 | 35 |

### sae_d200_k4_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 121 | 0.4576 | D1 | 24 |
| 72 | 0.4344 | D1 | 70 |
| 61 | 0.0451 | D1 | 40 |
| 148 | 0.0396 | D1 | 15 |
| 114 | 0.0331 | D1 | 51 |

### sae_d256_50000steps_k4_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 204 | 0.6014 | D1 | 24 |
| 223 | 0.2824 | D1 | 70 |
| 205 | 0.0310 | D1 | 28 |
| 132 | 0.0284 | D1 | 51 |
| 201 | 0.0260 | D1 | 63 |

### sae_d256_k4

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 204 | 0.6050 | D1 | 24 |
| 223 | 0.2877 | D1 | 70 |
| 132 | 0.0301 | D1 | 51 |
| 144 | 0.0289 | D1 | 94 |
| 205 | 0.0284 | D1 | 28 |

### sae_d256_k4_2layer_100dig_64d

| Feature | Firing Rate | Position | Best Digit |
|---------|-------------|----------|------------|
| 204 | 0.6050 | D1 | 24 |
| 223 | 0.2877 | D1 | 70 |
| 132 | 0.0301 | D1 | 51 |
| 144 | 0.0289 | D1 | 94 |
| 205 | 0.0284 | D1 | 28 |

## Analysis

- **L0**: Average number of active features per sample (lower = sparser)
- **Dead %**: Percentage of features that never fire (lower = better utilization)
- **MSE**: Mean squared reconstruction error (lower = better reconstruction)

### Key Observations

- Lowest dead feature %: **sae_d100_k4_50000steps_2layer_100dig_64d** (0.0%)
- Best reconstruction: **sae_d256_k4** (MSE: 0.0026)