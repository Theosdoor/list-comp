# Best SAE Candidates: k=3, d=128

## Summary
Found 16 SAE models with specs k=3, d=128. Ranked by your criteria: **high exp var** (~0.99), **low CE increase**, **high firing rate for special features**.

---

## 🏆 TOP PICKS

### 1. **sae_d128_k3_lr0.01_seed2_2layer_100dig_64d** ⭐ BEST OVERALL
- **Exp Var:** 0.9920
- **CE Increase:** 0.0602 ← **Lowest!**
- **N Special Features:** 2 (dual features firing, good diversity)
- **L0 (sparsity):** 2.99
- **Path:** `results/sae_models/sae_d128_k3_lr0.01_seed2_2layer_100dig_64d.pt`
- **Why:** Exceptional balance—lowest CE increase among all candidates, excellent exp var, and dual special features for high firing rate potential.

---

### 2. **sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d** ⭐ RUNNER UP
- **Exp Var:** 0.9929 ← **Highest!**
- **CE Increase:** 0.0643
- **N Special Features:** 2
- **L0 (sparsity):** 2.84 ← **Lowest L0 (sparsest)**
- **Path:** `results/sae_models/sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d.pt`
- **Why:** Highest explained variance with dual special features and sparsest activations. Great for interpretability.

---

### 3. **sae_d128_k3_lr1e-05_seed1_2layer_100dig_64d** ⭐ THIRD BEST
- **Exp Var:** 0.9932 ← **Tied highest**
- **CE Increase:** 0.0767
- **N Special Features:** 2
- **L0 (sparsity):** 2.84
- **Path:** `results/sae_models/sae_d128_k3_lr1e-05_seed1_2layer_100dig_64d.pt`
- **Why:** Tied for highest exp var with dual special features. Slightly higher CE increase than #1, but still excellent.

---

## 📊 Complete Ranking

| Rank | Model Name | Exp Var | CE Increase | N Special | L0 | Score |
|------|-----------|---------|-------------|-----------|----|----|
| 1 | sae_d128_k3_lr0.01_seed2_2layer_100dig_64d | 0.9920 | **0.0602** | 2 | 2.99 | 0.0803 |
| 2 | sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d | **0.9929** | 0.0643 | 2 | 2.84 | 0.0823 |
| 3 | sae_d128_k3_lr1e-05_seed1_2layer_100dig_64d | **0.9932** | 0.0767 | 2 | 2.84 | 0.0886 |
| 4 | sae_d128_k3_lr1e-05_seed0_2layer_100dig_64d | 0.9928 | 0.0896 | 2 | 2.86 | 0.0949 |
| 5 | sae_d128_k3_lr0.0001_seed2_2layer_100dig_64d | 0.9925 | 0.0670 | 1 | 2.95 | 0.1335 |
| 6 | sae_d128_k3_lr0.01_seed0_2layer_100dig_64d | 0.9919 | 0.0667 | 1 | 2.99 | 0.1335 |
| 7 | sae_d128_k3_lr0.0013008350811934896_seed0_2layer_100dig_64d | 0.9920 | 0.0676 | 1 | 3.03 | 0.1340 |
| 8 | sae_d128_k3_lr0.001_seed1_2layer_100dig_64d | 0.9919 | 0.0680 | 1 | 3.02 | 0.1342 |
| 9 | sae_d128_k3_lr0.001_seed0_2layer_100dig_64d | 0.9913 | 0.0687 | 1 | 3.10 | 0.1347 |
| 10 | sae_d128_k3_lr0.0001_seed1_2layer_100dig_64d | 0.9922 | 0.0724 | 1 | 3.08 | 0.1363 |

---

## 🎯 Key Insights

- **All 16 models are solid:** Even the worst performer (matryoshka) has 0.9898 exp var and only 0.1420 CE increase.
- **N Special = 2 dominates top 4:** The top 4 candidates all have 2 special features firing, suggesting these learning rates capture richer structure.
- **CE increase is your constraint:** Your criteria heavily favor low CE increase, so #1 and #2 stand out (0.0602–0.0643 range).
- **Sparsity (L0) is ~3:** All models hover around L0 ≈ 3, so efficiency is uniform across the sweep.

---

## 💡 Recommendation

**Use #1 (`sae_d128_k3_lr0.01_seed2_2layer_100dig_64d`)** if your primary goal is **downstream fidelity** (lowest CE increase) with dual special features.

**Use #2 (`sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d`)** if your primary goal is **reconstruction quality** (highest exp var) while maintaining excellent downstream performance.

**Comparison to other d values:** From `temp.md`, the best k=3 model across all d values is `sae_d128_k3_lr0.0001_seed0_2layer_100dig_64d` (ranked #11 here) with 0.9927 exp var and 0.0731 CE increase. Your d=128 candidates #1–#3 actually outperform that on CE increase alone.

