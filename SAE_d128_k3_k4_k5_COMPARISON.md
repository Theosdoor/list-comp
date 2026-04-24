# SAE Comparison: d=128, k=3 vs k=4 vs k=5
## Filtered: Exp Var > 0.99, CE Increase < 0.1

---

## 🏆 BEST OVERALL (All k values)

**`sae_d128_k5_lr0.0001_seed2_2layer_100dig_64d`** ← **WINNER**
- **k:** 5
- **Exp Var:** 0.9924
- **CE Increase:** **0.0585** ← Best CE across all k
- **Dead %:** **0.0%** ← Perfect dictionary!
- **N Special:** 2
- **Verdict:** Superior to k=3 best on CE (+0.0017 improvement) with zero dead features

---

## 📊 Summary Statistics

| Metric | K=3 | K=4 | K=5 |
|--------|-----|-----|-----|
| **Models passing filters** | 14 | 15 | 16 |
| **Best CE Increase** | 0.0602 | 0.0614 | **0.0585** |
| **Avg CE Increase** | 0.0712 | 0.0675 | **0.0658** |
| **Best Exp Var** | 0.9932 | 0.9928 | 0.9924 |
| **Avg Exp Var** | 0.9923 | 0.9921 | 0.9921 |
| **Lowest Dead %** | 3.1% | 0.0% | **0.0%** |
| **Avg Dead %** | 11.8% | 3.0% | **1.2%** |

---

## 🥇 Best by Category

### Best CE Increase (Downstream Fidelity)
**Winner: K=5** - `sae_d128_k5_lr0.0001_seed2_2layer_100dig_64d`
- CE: **0.0585**
- Dead %: 0.0%
- Improvement over k=3 best: -0.0017 (3% better)

### Best Exp Var (Reconstruction)
**Winner: K=3** - `sae_d128_k3_lr1e-05_seed1_2layer_100dig_64d`
- Exp Var: **0.9932**
- But: N=2 features, CE: 0.0767, Dead %: 12.5%

### Lowest Dead % (Cleanest Dictionary)
**Tie: K=4 & K=5** - Multiple models with **0.0%**
- K=5 examples: `sae_d128_k5_lr0.0001_seed0_2layer_100dig_64d` (CE: 0.0614, EV: 0.9923)
- K=4 examples: `sae_d128_k4_lr0.0001_seed1_2layer_100dig_64d` (CE: 0.0619, EV: 0.9925)

---

## 🔍 Top 5 Overall

| Rank | k | Model | Exp Var | CE Inc | Dead % |
|------|---|-------|---------|--------|--------|
| **1** | **5** | **sae_d128_k5_lr0.0001_seed2** | 0.9924 | **0.0585** | **0.0%** |
| 2 | 3 | sae_d128_k3_lr0.01_seed2 | 0.9920 | 0.0602 | 19.5% |
| 3 | 5 | sae_d128_k5_lr6.093... | 0.9923 | 0.0607 | 0.8% |
| 4 | 5 | sae_d128_k5_lr0.0001_seed0 | 0.9923 | 0.0614 | 0.0% |
| 5 | 4 | sae_d128_k4_lr0.00016... | 0.9922 | 0.0614 | 0.0% |

---

## 💡 Key Insights

### K=3 (14 passing)
- ✅ Good CE increase (0.0602 best)
- ✅ Highest exp var options (0.9932)
- ❌ **Higher dead % on average (11.8%)**
- ❌ Single features limit scope (10/14 are N=1)
- **Best for:** Clean mechanistic interpretation with single special feature

### K=4 (15 passing)
- ✅ **Excellent dead % reduction (3.0% avg)**
- ✅ Balanced CE increase (0.0675 avg)
- ✅ All models have 0% dead with perfect learning rates
- ⚠️ Fewer edge cases than k=3/k=5
- **Best for:** Balanced performance without extreme tuning

### K=5 (16 passing) 🏆
- ✅ **Best average CE increase (0.0658)** — 1% better than k=4
- ✅ **Minimal dead % (1.2% avg, many with 0%)**
- ✅ **100% dual features (N=2)** — richer mechanistic analysis
- ✅ Best overall CE: **0.0585** (top performer across all k)
- ✅ Most models passing filters (16/many attempted)
- ⚠️ Slightly lower avg exp var than k=3/k=4
- **Best for:** Highest downstream performance + clean activation dictionary

---

## 🎯 Recommendation by Priority

| Priority | Best Choice |
|----------|-------------|
| **Downstream CE** | K=5: `sae_d128_k5_lr0.0001_seed2` (CE: 0.0585) |
| **Clean dictionary** | K=5: `sae_d128_k5_lr0.0001_seed0` (Dead %: 0%, CE: 0.0614) |
| **Mechanistic simplicity** | K=3: `sae_d128_k3_lr0.01_seed0` (N=1, CE: 0.0667) |
| **Best balance** | K=5: `sae_d128_k5_lr0.0001_seed2` (CE + Dead % + EV) |
| **High exp var** | K=3: `sae_d128_k3_lr1e-05_seed1` (EV: 0.9932) |

---

## 📈 Performance Trends

### CE Increase Advantage (Lower is Better)
```
K=5 beats K=4: -1.7 bps (0.0658 vs 0.0675)
K=4 beats K=3: -3.7 bps (0.0675 vs 0.0712)
K=5 beats K=3: -5.4 bps (0.0658 vs 0.0712)
```

### Dead % Advantage (Lower is Better)
```
K=5 beats K=4: -1.8 pts (1.2% vs 3.0%)
K=4 beats K=3: -8.8 pts (3.0% vs 11.8%)
K=5 beats K=3: -10.6 pts (1.2% vs 11.8%)
```

**Conclusion:** K=5 dominates on both metrics while K=3 excels only in edge cases (specific mechanistic needs, N=1 constraints).

---

## 📌 Final Ranking by Use Case

### Absolute Best (No Constraints)
→ **K=5: `sae_d128_k5_lr0.0001_seed2`**
- Best CE: 0.0585
- Perfect Dead %: 0.0%
- Solid Exp Var: 0.9924
- Dual features for rich analysis

### Best if You Need Clean Single Feature (N=1)
→ **K=3: `sae_d128_k3_lr0.01_seed0`**
- CE: 0.0667 (competitive)
- N=1 mechanistic clarity
- Trade-off: 17.2% dead (acceptable for interpretability)

### Best if Exp Var is Priority
→ **K=3: `sae_d128_k3_lr1e-05_seed1`**
- Exp Var: 0.9932 (highest)
- CE: 0.0767 (acceptable)
- N=2 features
- Trade-off: 12.5% dead

### Safe Middle Ground
→ **K=4: `sae_d128_k4_lr0.0001_seed1`**
- CE: 0.0619
- Dead %: 0.0%
- Exp Var: 0.9925
- All excellent metrics, fewer extreme values

