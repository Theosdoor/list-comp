# Best SAEs Analysis

Based on the 1870 models evaluated on the list-copy task. Metric key:
- **LR**: Loss Recovered (higher is better)
- **L0**: Actual L0 / mean features active per token (lower is better)
- **Dead**: Percentage of dead features (lower is better)

## Top 5 Overall SAEs (Ranked)

1. `jumprelu_sae_d256_tl05_sp5_lr7e-05_seed3_2layer_100dig_64d`
   - **LR:** 0.9998 | **L0:** 4.03 | **Dead:** 50.8%
2. `jumprelu_sae_d320_tl04_sp1_lr0.0003_seed4_best_2layer_100dig_64d`
   - **LR:** 0.9998 | **L0:** 3.69 | **Dead:** 54.7%
3. `jumprelu_sae_d192_tl04_sp1_lr7e-05_seed8_2layer_100dig_64d`
   - **LR:** 0.9998 | **L0:** 3.09 | **Dead:** 39.6%
4. `jumprelu_sae_d192_tl04_sp1_lr7e-05_seed6_2layer_100dig_64d`
   - **LR:** 0.9998 | **L0:** 3.09 | **Dead:** 39.6%
5. `jumprelu_sae_d192_tl04_sp1_lr7e-05_seed5_2layer_100dig_64d`
   - **LR:** 0.9998 | **L0:** 3.09 | **Dead:** 39.6%

> [!NOTE]
> **Insights**: The overall top 5 models are unanimously JumpReLU SAEs with high {sae}$ (192). They completely max out the Loss Recovered metric at 0.9998 while keeping actual $L_0$ sparse (~4.0). Noticeably, their dead feature percentages are quite high (~38-40%), showing that perfect task recovery doesn't require full dictionary utilization.

## JUMPRELU SAEs
### Top 3 by Target $ / $ (Optimized for Loss Recovered)

**Target $ / $ = 1**
- `jumprelu_sae_d192_tl01_sp0.1_lr7e-05_seed9_2layer_100dig_64d` | LR: 0.9995 | L0: 2.84 | Dead: 42.2% | d_sae: 192
- `jumprelu_sae_d192_tl01_sp0.1_lr7e-05_seed8_2layer_100dig_64d` | LR: 0.9995 | L0: 2.84 | Dead: 42.2% | d_sae: 192
- `jumprelu_sae_d192_tl01_sp0.1_lr7e-05_seed7_2layer_100dig_64d` | LR: 0.9995 | L0: 2.84 | Dead: 42.2% | d_sae: 192

**Target $ / $ = 2**
- `jumprelu_sae_d320_tl02_sp0.1_lr7e-05_seed4_2layer_100dig_64d` | LR: 0.9997 | L0: 3.07 | Dead: 54.4% | d_sae: 320
- `jumprelu_sae_d192_tl02_sp0.1_lr7e-05_seed4_2layer_100dig_64d` | LR: 0.9997 | L0: 2.91 | Dead: 40.1% | d_sae: 192
- `jumprelu_sae_d192_tl02_sp0.1_lr7e-05_seed2_2layer_100dig_64d` | LR: 0.9997 | L0: 2.91 | Dead: 40.1% | d_sae: 192

**Target $ / $ = 3**
- `jumprelu_sae_d192_tl03_sp1_lr7e-05_seed2_2layer_100dig_64d` | LR: 0.9997 | L0: 3.00 | Dead: 42.2% | d_sae: 192
- `jumprelu_sae_d192_tl03_sp0.1_lr7e-05_seed8_2layer_100dig_64d` | LR: 0.9997 | L0: 3.05 | Dead: 42.7% | d_sae: 192
- `jumprelu_sae_d192_tl03_sp1_lr7e-05_seed8_2layer_100dig_64d` | LR: 0.9997 | L0: 3.01 | Dead: 42.2% | d_sae: 192

**Target $ / $ = 4**
- `jumprelu_sae_d320_tl04_sp1_lr0.0003_seed4_best_2layer_100dig_64d` | LR: 0.9998 | L0: 3.69 | Dead: 54.7% | d_sae: 320
- `jumprelu_sae_d192_tl04_sp5_lr7e-05_seed7_2layer_100dig_64d` | LR: 0.9998 | L0: 3.07 | Dead: 39.6% | d_sae: 192
- `jumprelu_sae_d192_tl04_sp1_lr7e-05_seed9_2layer_100dig_64d` | LR: 0.9998 | L0: 3.09 | Dead: 39.6% | d_sae: 192

**Target $ / $ = 5**
- `jumprelu_sae_d192_tl05_sp5_lr7e-05_seed6_2layer_100dig_64d` | LR: 0.9998 | L0: 4.18 | Dead: 36.5% | d_sae: 192
- `jumprelu_sae_d128_tl05_lr7e-05_seed0_2layer_100dig_64d` | LR: 0.9998 | L0: 3.24 | Dead: 18.0% | d_sae: 128
- `jumprelu_sae_d192_tl05_sp5_lr7e-05_seed8_2layer_100dig_64d` | LR: 0.9998 | L0: 4.18 | Dead: 36.5% | d_sae: 192

### Best 3 Optimizing for Dead % (Fewest Dead Features)
- `jumprelu_sae_d128_tl05_sp5_lr7e-05_seed10_2layer_100dig_64d` | LR: 0.9998 | L0: 3.18 | Dead: 15.6% | d_sae: 128
- `jumprelu_sae_d128_tl05_sp5_lr7e-05_seed11_2layer_100dig_64d` | LR: 0.9998 | L0: 3.18 | Dead: 15.6% | d_sae: 128
- `jumprelu_sae_d128_tl05_sp5_lr7e-05_seed12_2layer_100dig_64d` | LR: 0.9998 | L0: 3.18 | Dead: 15.6% | d_sae: 128

### Best 3 Optimizing for Loss Recovered
- `jumprelu_sae_d192_tl04_sp5_lr7e-05_seed5_2layer_100dig_64d` | LR: 0.9998 | L0: 3.07 | Dead: 39.6% | d_sae: 192
- `jumprelu_sae_d192_tl04_sp5_lr7e-05_seed6_2layer_100dig_64d` | LR: 0.9998 | L0: 3.07 | Dead: 39.6% | d_sae: 192
- `jumprelu_sae_d192_tl04_sp5_lr7e-05_seed7_2layer_100dig_64d` | LR: 0.9998 | L0: 3.07 | Dead: 39.6% | d_sae: 192

### Best 3 Optimizing for Actual $ (Sparsest models with LR > 0.95)
- `jumprelu_sae_d256_tl02_sp5_lr0.0003_seed4_2layer_100dig_64d` | LR: 0.9910 | L0: 2.32 | Dead: 59.8% | d_sae: 256
- `jumprelu_sae_d256_tl02_sp5_lr0.0003_seed3_2layer_100dig_64d` | LR: 0.9910 | L0: 2.32 | Dead: 59.8% | d_sae: 256
- `jumprelu_sae_d256_tl02_sp5_lr0.0003_seed2_2layer_100dig_64d` | LR: 0.9910 | L0: 2.32 | Dead: 59.8% | d_sae: 256

## BTK SAEs
### Top 3 by Target $ / $ (Optimized for Loss Recovered)

**Target $ / $ = 1**
- `btk_sae_d192_k1_lr0.0003_seed6_2layer_100dig_64d` | LR: 0.9431 | L0: 1.00 | Dead: 46.9% | d_sae: 192
- `btk_sae_d192_k1_lr0.0003_seed3_2layer_100dig_64d` | LR: 0.9429 | L0: 1.00 | Dead: 45.3% | d_sae: 192
- `btk_sae_d192_k1_lr0.0003_seed7_2layer_100dig_64d` | LR: 0.9427 | L0: 1.00 | Dead: 44.3% | d_sae: 192

**Target $ / $ = 2**
- `btk_sae_d320_k2_lr0.0003_seed8_2layer_100dig_64d` | LR: 0.9853 | L0: 2.11 | Dead: 60.9% | d_sae: 320
- `btk_sae_d128_k2_lr0.0003_seed2_2layer_100dig_64d` | LR: 0.9841 | L0: 1.96 | Dead: 15.6% | d_sae: 128
- `btk_sae_d192_k2_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9840 | L0: 1.95 | Dead: 42.2% | d_sae: 192

**Target $ / $ = 3**
- `btk_sae_d192_k3_lr0.0003_seed5_2layer_100dig_64d` | LR: 0.9997 | L0: 3.36 | Dead: 26.0% | d_sae: 192
- `btk_sae_d320_k3_lr0.0003_seed13_2layer_100dig_64d` | LR: 0.9997 | L0: 2.99 | Dead: 56.9% | d_sae: 320
- `btk_sae_d256_k3_lr0.0003_seed10_2layer_100dig_64d` | LR: 0.9997 | L0: 2.93 | Dead: 36.7% | d_sae: 256

**Target $ / $ = 4**
- `btk_sae_d256_k4_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9997 | L0: 3.72 | Dead: 10.9% | d_sae: 256
- `btk_sae_d320_k4_lr0.0003_seed8_2layer_100dig_64d` | LR: 0.9997 | L0: 3.90 | Dead: 11.2% | d_sae: 320
- `btk_sae_d256_k4_lr0.0003_seed3_2layer_100dig_64d` | LR: 0.9997 | L0: 4.76 | Dead: 13.3% | d_sae: 256

**Target $ / $ = 5**
- `btk_sae_d320_k5_lr0.0003_seed2_2layer_100dig_64d` | LR: 0.9997 | L0: 4.97 | Dead: 9.7% | d_sae: 320
- `btk_sae_d320_k5_lr0.0003_seed0_2layer_100dig_64d` | LR: 0.9997 | L0: 4.82 | Dead: 14.1% | d_sae: 320
- `btk_sae_d256_k5_lr0.0003_seed9_2layer_100dig_64d` | LR: 0.9997 | L0: 4.80 | Dead: 5.1% | d_sae: 256

### Best 3 Optimizing for Dead % (Fewest Dead Features)
- `btk_sae_d192_k5_lr0.0003_seed13_2layer_100dig_64d` | LR: 0.9997 | L0: 4.80 | Dead: 0.0% | d_sae: 192
- `btk_sae_d128_k4_lr0.0003_seed4_2layer_100dig_64d` | LR: 0.9996 | L0: 3.83 | Dead: 0.0% | d_sae: 128
- `btk_sae_d128_k4_lr0.0003_seed11_2layer_100dig_64d` | LR: 0.9996 | L0: 4.00 | Dead: 0.0% | d_sae: 128

### Best 3 Optimizing for Loss Recovered
- `btk_sae_d320_k3_lr0.0003_seed0_2layer_100dig_64d` | LR: 0.9997 | L0: 2.88 | Dead: 41.9% | d_sae: 320
- `btk_sae_d256_k3_lr0.0003_seed10_2layer_100dig_64d` | LR: 0.9997 | L0: 2.93 | Dead: 36.7% | d_sae: 256
- `btk_sae_d256_k3_lr0.0003_seed13_2layer_100dig_64d` | LR: 0.9997 | L0: 2.98 | Dead: 56.6% | d_sae: 256

### Best 3 Optimizing for Actual $ (Sparsest models with LR > 0.95)
- `btk_sae_d128_k2_lr0.0003_seed1_2layer_100dig_64d` | LR: 0.9829 | L0: 1.93 | Dead: 19.5% | d_sae: 128
- `btk_sae_d128_k2_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9830 | L0: 1.93 | Dead: 17.2% | d_sae: 128
- `btk_sae_d256_k2_lr0.0003_seed5_2layer_100dig_64d` | LR: 0.9818 | L0: 1.93 | Dead: 53.9% | d_sae: 256

## MATRYOSHKA SAEs
### Top 3 by Target $ / $ (Optimized for Loss Recovered)

**Target $ / $ = 1**
- `matryoshka_sae_d128_k1_ng2_lr0.0003_seed3_2layer_100dig_64d` | LR: 0.9392 | L0: 1.00 | Dead: 11.7% | d_sae: 128
- `matryoshka_sae_d128_k1_ng3_lr0.0003_seed7_2layer_100dig_64d` | LR: 0.9377 | L0: 1.00 | Dead: 11.7% | d_sae: 128
- `matryoshka_sae_d128_k1_ng2_lr0.0003_seed7_2layer_100dig_64d` | LR: 0.9374 | L0: 1.00 | Dead: 14.8% | d_sae: 128

**Target $ / $ = 2**
- `matryoshka_sae_d128_k2_ng3_lr0.0003_seed3_2layer_100dig_64d` | LR: 0.9765 | L0: 2.00 | Dead: 17.2% | d_sae: 128
- `matryoshka_sae_d128_k2_ng3_lr0.0003_seed9_2layer_100dig_64d` | LR: 0.9764 | L0: 2.00 | Dead: 12.5% | d_sae: 128
- `matryoshka_sae_d128_k2_ng3_lr0.0003_seed8_2layer_100dig_64d` | LR: 0.9763 | L0: 2.00 | Dead: 20.3% | d_sae: 128

**Target $ / $ = 3**
- `matryoshka_sae_d128_k3_ng1_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9997 | L0: 2.88 | Dead: 7.8% | d_sae: 128
- `matryoshka_sae_d320_k3_ng2_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9997 | L0: 3.06 | Dead: 40.0% | d_sae: 320
- `matryoshka_sae_d256_k3_ng1_lr0.0003_seed11_2layer_100dig_64d` | LR: 0.9997 | L0: 3.07 | Dead: 36.3% | d_sae: 256

**Target $ / $ = 4**
- `matryoshka_sae_d192_k4_ng1_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9997 | L0: 3.70 | Dead: 2.1% | d_sae: 192
- `matryoshka_sae_d320_k4_ng1_lr0.0003_seed0_2layer_100dig_64d` | LR: 0.9997 | L0: 3.77 | Dead: 11.2% | d_sae: 320
- `matryoshka_sae_d320_k4_ng2_lr0.0003_seed8_2layer_100dig_64d` | LR: 0.9997 | L0: 3.87 | Dead: 4.7% | d_sae: 320

**Target $ / $ = 5**
- `matryoshka_sae_d320_k5_ng2_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9997 | L0: 4.75 | Dead: 2.5% | d_sae: 320
- `matryoshka_sae_d320_k5_ng2_lr0.0003_seed1_2layer_100dig_64d` | LR: 0.9997 | L0: 4.93 | Dead: 0.9% | d_sae: 320
- `matryoshka_sae_d320_k5_ng1_lr0.0003_seed11_2layer_100dig_64d` | LR: 0.9997 | L0: 4.82 | Dead: 9.1% | d_sae: 320

### Best 3 Optimizing for Dead % (Fewest Dead Features)
- `matryoshka_sae_d128_k4_ng1_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9996 | L0: 3.78 | Dead: 0.0% | d_sae: 128
- `matryoshka_sae_d128_k4_ng1_lr0.0003_seed14_2layer_100dig_64d` | LR: 0.9996 | L0: 3.80 | Dead: 0.0% | d_sae: 128
- `matryoshka_sae_d128_k4_ng1_lr0.0003_seed2_2layer_100dig_64d` | LR: 0.9996 | L0: 3.86 | Dead: 0.0% | d_sae: 128

### Best 3 Optimizing for Loss Recovered
- `matryoshka_sae_d256_k3_ng2_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9997 | L0: 2.86 | Dead: 29.3% | d_sae: 256
- `matryoshka_sae_d128_k3_ng1_lr0.0003_seed12_2layer_100dig_64d` | LR: 0.9997 | L0: 2.88 | Dead: 7.8% | d_sae: 128
- `matryoshka_sae_d192_k3_ng1_lr0.0003_seed1_2layer_100dig_64d` | LR: 0.9997 | L0: 2.91 | Dead: 24.5% | d_sae: 192

### Best 3 Optimizing for Actual $ (Sparsest models with LR > 0.95)
- `matryoshka_sae_d128_k2_ng3_lr0.0003_seed4_2layer_100dig_64d` | LR: 0.9679 | L0: 2.00 | Dead: 14.8% | d_sae: 128
- `matryoshka_sae_d128_k2_ng2_lr0.0003_seed11_2layer_100dig_64d` | LR: 0.9681 | L0: 2.00 | Dead: 11.7% | d_sae: 128
- `matryoshka_sae_d128_k2_ng3_lr0.0003_seed9_2layer_100dig_64d` | LR: 0.9764 | L0: 2.00 | Dead: 12.5% | d_sae: 128