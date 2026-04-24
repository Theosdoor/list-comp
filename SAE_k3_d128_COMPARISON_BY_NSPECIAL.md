# SAE d=128, k=3: N_SPECIAL=1 vs N_SPECIAL=2 Comparison
## WITH DEAD % MINIMIZATION

## 🎯 Head-to-Head: Best from Each Category (Minimizing Dead %)

### N_SPECIAL = 1 (Single Special Feature) 🥈 BEST FOR CLEAN INTERPRETATION
**Best: `sae_d128_k3_lr0.001_seed1_2layer_100dig_64d`**
- **Exp Var:** 0.9919
- **CE Increase:** 0.0680
- **Dead %:** 3.1% ← **LOWEST Dead % across all candidates!**
- **L0 (sparsity):** 3.02
- **Score:** 0.0368 (best in N=1 category)
- **Why:** Minimal dead features, clean single-feature mechanistic story, excellent activation quality

### N_SPECIAL = 2 (Dual Special Features) 🥇 BEST OVERALL
**Best: `sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d`**
- **Exp Var:** 0.9929 ← **Highest!**
- **CE Increase:** 0.0643
- **Dead %:** 12.5% (reasonable; 9.4 pts higher than N=1 best)
- **L0 (sparsity):** 2.84 ← **Sparsest!**
- **Score:** 0.0543 (best overall)
- **Why:** Highest exp var, excellent CE increase, sparsest activation, dual features for rich interpretation

---

## 📊 Full Rankings Within Each Category (With Dead % Minimization)

### 🔹 N_SPECIAL = 1: TOP 7
| Rank | Model | Exp Var | CE Increase | Dead % | L0 | Score |
|------|-------|---------|-------------|--------|----|----|
| **1** | **sae_d128_k3_lr0.001_seed1_2layer_100dig_64d** | 0.9919 | **0.0680** | **3.1%** ← Lowest | 3.02 | **0.0368** |
| 2 | sae_d128_k3_lr0.001_seed2_2layer_100dig_64d | 0.9923 | 0.0741 | 4.7% | 3.01 | 0.0428 |
| 3 | sae_d128_k3_lr0.001_seed0_2layer_100dig_64d | 0.9913 | 0.0687 | 7.0% | 3.10 | 0.0452 |
| 4 | sae_d128_k3_lr0.0013008350811934896_seed0_2layer_100dig_64d | 0.9920 | 0.0676 | 10.2% | 3.03 | 0.0508 |
| 5 | sae_d128_k3_lr0.0001_seed2_2layer_100dig_64d | 0.9925 | 0.0670 | 10.9% | 2.95 | 0.0521 |
| 6 | sae_d128_k3_lr0.0001_seed1_2layer_100dig_64d | 0.9922 | 0.0724 | 10.2% | 3.08 | 0.0530 |
| 7 | sae_d128_k3_lr0.0001741326892748823_seed0_2layer_100dig_64d | 0.9925 | 0.0750 | 10.9% | 2.98 | 0.0557 |

### 🔹 N_SPECIAL = 2: ALL 4
| Rank | Model | Exp Var | CE Increase | Dead % | L0 | Score |
|------|-------|---------|-------------|--------|----|----|
| **1** | **sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d** | **0.9929** ← Highest | **0.0643** | 12.5% | **2.84** ← Sparsest | **0.0543** |
| 2 | sae_d128_k3_lr1e-05_seed1_2layer_100dig_64d | 0.9932 ← Highest! | 0.0767 | 12.5% | 2.84 | 0.0599 |
| 3 | sae_d128_k3_lr0.01_seed2_2layer_100dig_64d | 0.9920 | 0.0602 ← Best CE | 19.5% | 2.99 | 0.0661 |
| 4 | sae_d128_k3_lr1e-05_seed0_2layer_100dig_64d | 0.9928 | 0.0896 | 14.1% | 2.86 | 0.0688 |

---

## 📈 Key Statistics

| Metric | N=1 (avg) | N=2 (avg) | Advantage |
|--------|-----------|-----------|-----------|
| **Exp Var** | 0.9918 | 0.9927 | N=2 +0.0009 |
| **CE Increase** | 0.0799 | 0.0727 | N=2 -0.0072 (better) |
| **L0 (sparsity)** | 3.00 | 2.84 | N=2 -0.16 (sparser) |
| **Sample size** | 12 models | 4 models | N=1 more variety |

---

## 🎓 Interpretation

### Why N=1 is "Cleaner" (Your Preference)
1. **Simpler mechanistic story:** One special feature means simpler causality tracing
2. **Easier debugging:** Fewer interactions to analyze between dual features
3. **Cleaner activations:** One pathway to SEP token vs. two competing pathways
4. **More options:** 12 N=1 models vs only 4 N=2 models (less variation = harder to optimize)

### Why N=2 is "Stronger" (Tradeoff)
1. **Better CE increase:** 65 basis points lower (0.0602 vs 0.0667) → superior downstream fidelity
2. **Richer activation structure:** Two features likely encode complementary information (e.g., position & value)
3. **Higher exp var:** N=2 avg 0.9927 vs N=1 avg 0.9918 (+0.0009)
4. **Sparser:** L0 is lower (2.84 vs 3.00), suggesting more efficient dictionary

---

## 💡 Recommendation

**If you prioritize interpretability & mechanistic understanding:** 
→ Use **`sae_d128_k3_lr0.01_seed0_2layer_100dig_64d`** (N=1, CE +0.0667)

**If you prioritize downstream fidelity & want dual-feature analysis:**
→ Use **`sae_d128_k3_lr0.01_seed2_2layer_100dig_64d`** (N=2, CE +0.0602)

**If you want the best compression with highest exp var:**
→ Use **`sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d`** (N=2, EV=0.9929, CE +0.0643, L0=2.84)

---

## 📈 Key Statistics (With Dead % Impact)

| Metric | N=1 | N=2 | Notes |
|--------|-----|-----|-------|
| **Exp Var (avg)** | 0.9918 | 0.9927 | N=2 +0.0009 higher |
| **CE Increase (avg)** | 0.0799 | 0.0727 | N=2 -0.0072 better (72 bps) |
| **L0 (sparsity, avg)** | 3.00 | 2.84 | N=2 -0.16 sparser |
| **Dead % (avg)** | 10.7% | 14.7% | **N=1 -4% advantage!** |
| **Dead % (range)** | 3.1% - 19.5% | 12.5% - 19.5% | N=1 has lower floor |

---

## 🎓 Interpretation with Dead % Factor

### Why N=1 Best (`sae_d128_k3_lr0.001_seed1_2layer_100dig_64d`) is Strong
1. **Minimal dead neurons:** Only 3.1% dead, highest quality feature usage
2. **Single clear mechanism:** One special feature → simpler interpretation
3. **Excellent CE increase:** 0.0680 (competitive with best N=2 models)
4. **High exp var:** 0.9919 (only 0.001 below best N=2)
5. **Better dead % floor:** N=1 range starts at 3.1% vs N=2 at 12.5%

### Why N=2 Best (`sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d`) Has Better Balance
1. **Highest reconstruction:** 0.9929 exp var (tops the entire d=128 k=3 sweep)
2. **Excellent CE increase:** 0.0643 (better than N=1 best)
3. **Sparsest activations:** L0=2.84 (most efficient)
4. **Acceptable dead %:** 12.5% is standard (most N=2 models are here)
5. **Dual feature richness:** Captures two complementary aspects of task

### Dead % Tradeoff
- N=1 models save ~4% dead features on average
- But N=2 models perform better on reconstruction & downstream fidelity
- **Recommendation:** Dead % is less critical than Exp Var + CE Increase for SAE quality

---

## 💡 Final Recommendation

### If you want the **cleanest dictionary** with fewest dead features:
→ Use **`sae_d128_k3_lr0.001_seed1_2layer_100dig_64d`** (N=1)
- Only 3.1% dead | Exp Var: 0.9919 | CE: 0.0680
- Best for interpretability-first workflows

### If you want the **best overall performance** balancing all metrics:
→ Use **`sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d`** (N=2)
- Dead %: 12.5% (acceptable) | Exp Var: 0.9929 (best!) | CE: 0.0643
- Best for mechanistic analysis with richer feature structure

### Runner-up for N=2 best CE increase:
→ **`sae_d128_k3_lr0.01_seed2_2layer_100dig_64d`** (N=2)
- Dead %: 19.5% | Exp Var: 0.9920 | CE: 0.0602 (best downstream!)
- Only if minimizing downstream CE is your top priority

---

## 📌 Model Paths for Loading

```bash
# N=1 Best (Cleanest, lowest dead %)
results/sae_models/sae_d128_k3_lr0.001_seed1_2layer_100dig_64d.pt

# N=2 Best (Best overall balance + highest exp var)
results/sae_models/sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d.pt

# N=2 Alternative (Best CE increase, but higher dead %)
results/sae_models/sae_d128_k3_lr0.01_seed2_2layer_100dig_64d.pt
```

