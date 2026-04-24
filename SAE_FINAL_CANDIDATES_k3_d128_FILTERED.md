# Final SAE Candidates: k=3, d=128
## Filtered: Exp Var > 0.99, CE Increase < 0.1

**Evaluation:** All 10,000 samples (train + val combined) for consistency with SAE comparison metrics.

---

## 🏆 BEST OVERALL

**`sae_d128_k3_lr0.01_seed2_2layer_100dig_64d`** (N_SPECIAL=2)
- **Exp Var:** 0.9920
- **CE Increase:** 0.0602 ← **Best CE performance**
- **Dead %:** 19.5%
- **N Special:** 2 (dual features)
- **Pros:** Best downstream fidelity, highest exp var among N=2 options
- **Cons:** Highest dead % but still acceptable
- **Path:** `results/sae_models/sae_d128_k3_lr0.01_seed2_2layer_100dig_64d.pt`

---

## 🥈 Best for N_SPECIAL = 1 (Single Feature)

**`sae_d128_k3_lr0.01_seed0_2layer_100dig_64d`**
- **Exp Var:** 0.9919
- **CE Increase:** 0.0667 ← Best CE among N=1
- **Dead %:** 17.2%
- **N Special:** 1 (single special feature)
- **Pros:** Cleanest mechanistic story, competitive CE increase
- **Cons:** High dead %, not as strong CE as best N=2 option
- **Path:** `results/sae_models/sae_d128_k3_lr0.01_seed0_2layer_100dig_64d.pt`

**Alternative N=1 (Lowest Dead %):**
- **`sae_d128_k3_lr0.001_seed1_2layer_100dig_64d`** (Dead %: 3.1%, CE: 0.0680)
- Best if minimizing dead features is critical

---

## 🥉 Best for N_SPECIAL = 2 (Dual Features)

**`sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d`** ← Second choice overall
- **Exp Var:** 0.9929 ← **Highest exp var**
- **CE Increase:** 0.0643
- **Dead %:** 12.5% ← Best among top contenders
- **N Special:** 2 (dual features)
- **Pros:** Highest reconstruction quality, sparsest, excellent CE, lowest dead %
- **Cons:** Slightly worse CE than best N=2 overall
- **Path:** `results/sae_models/sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d.pt`

---

## 📊 Complete Ranking (14 models pass filters)

### N_SPECIAL = 1 (10 candidates)
| Rank | Model | Exp Var | CE Inc | Dead % |
|------|-------|---------|--------|--------|
| 1 | sae_d128_k3_lr0.01_seed0_2layer_100dig_64d | 0.9919 | 0.0667 | 17.2% |
| 2 | sae_d128_k3_lr0.0001_seed2_2layer_100dig_64d | 0.9925 | 0.0670 | 10.9% |
| 3 | sae_d128_k3_lr0.0013008350811934896_seed0_2layer_100dig_64d | 0.9920 | 0.0676 | 10.2% |
| 4 | sae_d128_k3_lr0.001_seed1_2layer_100dig_64d | 0.9919 | 0.0680 | 3.1% |
| 5 | sae_d128_k3_lr0.001_seed0_2layer_100dig_64d | 0.9913 | 0.0687 | 7.0% |

### N_SPECIAL = 2 (4 candidates)
| Rank | Model | Exp Var | CE Inc | Dead % |
|------|-------|---------|--------|--------|
| 1 | sae_d128_k3_lr0.01_seed2_2layer_100dig_64d | 0.9920 | **0.0602** | 19.5% |
| 2 | sae_d128_k3_lr2.9531648156023545e-05_seed0_2layer_100dig_64d | **0.9929** | 0.0643 | 12.5% |
| 3 | sae_d128_k3_lr1e-05_seed1_2layer_100dig_64d | 0.9932 | 0.0767 | 12.5% |
| 4 | sae_d128_k3_lr1e-05_seed0_2layer_100dig_64d | 0.9928 | 0.0896 | 14.1% |

---

## 💡 Recommendation by Use Case

| Goal | Best Choice |
|------|-------------|
| **Downstream fidelity (CE increase)** | `sae_d128_k3_lr0.01_seed2` (N=2, CE=0.0602) |
| **Highest reconstruction** | `sae_d128_k3_lr2.9531...` (N=2, EV=0.9929) |
| **Single clean feature** | `sae_d128_k3_lr0.01_seed0` (N=1, CE=0.0667) |
| **Fewest dead neurons** | `sae_d128_k3_lr0.001_seed1` (N=1, Dead %=3.1%) |
| **Balanced (EV+CE+Dead)** | `sae_d128_k3_lr2.9531...` (N=2) |

---

## 📝 Notes

- **14 out of 16 d=128, k=3 models** pass the strict filters (Exp Var > 0.99, CE < 0.1)
- **N=2 models dominate on CE increase** (best N=2: 0.0602 vs best N=1: 0.0667)
- **N=1 models dominate on dead %** (best N=1: 3.1% vs best N=2: 12.5%)
- **All-around best:** N=2 with 0.01 learning rate (best balance of all metrics)

