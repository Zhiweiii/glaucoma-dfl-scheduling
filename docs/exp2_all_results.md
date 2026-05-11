# Exp 2 — All Results Summary

Self-contained reference for analysis and plotting.
All results are from the test set (N=333, K=[16, 33, 66]).

**Branch:** `single-head-framework`  
**Data dir:** `/data/lizhiwei/dfl_v2/v5/exp2/`  
**Availability:** r5, p=[0.25, 0.50, 1.00] per slot, seeds train=0 / val=100 / test=200

---

## Setup

### Cost function
| Param | Value |
|-------|-------|
| `alpha` (severity costs) | [0, 1, 3, 6, 10] for grades 0–4 |
| `beta` (per-referral) | 0.5 |
| `delay` (slot weights) | [1.0, 3.0, 8.0] |
| `d_miss` (miss penalty) | 15.0 |
| `K_frac_list` (slot capacities) | [0.05, 0.10, 0.20] → K=[16, 33, 66] |

### Metrics
- **C_norm**: `C_total / C_random`. Lower is better. Oracle = 0.780 (best achievable with perfect labels), Random = 1.000 (baseline).
- **C_total**: raw scheduling cost on test set
- **C_oracle**: optimal cost with true severity labels known (lower bound), solved by Gurobi ILP
- **C_random**: expected cost under random triage scores (MC average over 1000 draws), using same availability-constrained solver
- **C_regret**: `C_total − C_oracle` (gap from oracle)
- **recall@K**: fraction of true grade≥3 patients scheduled in **any** slot
- **pairwise_accuracy**: fraction of grade-discordant pairs (Y_i ≠ Y_j) where model ranking agrees with label ordering
- **auc_roc**: AUC-ROC of triage score for separating **grade≥3 (severe) vs grade 1–2 (mild)**

### Methods
| Method | Description |
|--------|-------------|
| M1 | Binary triage only (VGG19 → trunk → binary_head). Triage score = P(glaucoma). |
| M2 | Severity CE fine-tune from M1. Checkpoint by val CE loss. |
| M3_old | Severity CE fine-tune. Checkpoint by val scheduling cost (single availability realization). |
| M3_fix | Severity CE fine-tune. Checkpoint by val cost averaged over 5 availability realizations. |
| M4_sigma* | M3 Stage 2 (CE) + Stage 3 DFL (REINFORCE). σ = perturbation std in randomised smoothing. |

**Architecture (all methods):** VGG19 backbone → trunk (Linear 25088→64→128) → severity_head (Linear 128→5).  
Triage score: `α̂_i = Σ_k α_k · softmax(sev_logits_i)_k ∈ [0, 10]`.  
M2/M3/M4 unfreeze backbone layers 9+ + trunk + severity_head (21.4M trainable params) from M1 init.

---

## Raw Results (per seed)

```csv
method,seed,C_norm,C_total,C_oracle,C_random,C_regret,recall_at_K,pairwise_accuracy,auc_roc,note
M1,42,0.9619,19411.5,15742.5,20180.422,3669.0,0.3777,0.5614,0.5478,Binary triage only
M1,43,0.9764,19703.5,15742.5,20180.422,3961.0,0.3670,0.5383,0.5273,Binary triage only
M1,44,0.9739,19654.5,15742.5,20180.422,3912.0,0.3617,0.5344,0.5099,Binary triage only
M2,42,0.8900,17961.5,15742.5,20180.422,2219.0,0.4840,0.6949,0.7166,Severity CE / val CE checkpoint
M2,43,0.8998,18157.5,15742.5,20180.422,2415.0,0.4734,0.6873,0.7028,Severity CE / val CE checkpoint
M2,44,0.8974,18109.5,15742.5,20180.422,2367.0,0.4734,0.6810,0.6924,Severity CE / val CE checkpoint
M3_old,42,0.8900,,,,,0.4840,0.6949,0.7166,Val cost checkpoint (1 realization)
M3_old,43,0.9145,,,,,0.4574,0.6687,0.6824,Val cost checkpoint (1 realization)
M3_old,44,0.9083,,,,,0.4574,0.6748,0.6827,Val cost checkpoint (1 realization)
M3_fix,42,0.8900,17961.5,15742.5,20180.422,2219.0,0.4840,0.6949,0.7166,Val cost checkpoint (5 realizations)
M3_fix,43,0.9098,18359.5,15742.5,20180.422,2617.0,0.4628,0.6753,0.6870,Val cost checkpoint (5 realizations)
M3_fix,44,0.9015,18192.5,15742.5,20180.422,2450.0,0.4681,0.6750,0.6858,Val cost checkpoint (5 realizations)
M4_sigma0.5,42,0.9064,18292.5,15742.5,20180.422,2550.0,0.4681,0.6763,0.6842,DFL sigma=0.5; Stage3 fails all seeds
M4_sigma0.5,43,0.8871,17901.5,15742.5,20180.422,2159.0,0.4947,0.6928,0.7060,DFL sigma=0.5; Stage3 fails all seeds
M4_sigma0.5,44,0.8991,18143.5,15742.5,20180.422,2401.0,0.4787,0.6828,0.6914,DFL sigma=0.5; Stage3 fails all seeds
M4_sigma1.0,42,0.9064,18292.5,15742.5,20180.422,2550.0,0.4681,0.6763,0.6842,DFL sigma=1.0; Stage3 fails all seeds
M4_sigma1.0,43,0.8871,17901.5,15742.5,20180.422,2159.0,0.4947,0.6928,0.7060,DFL sigma=1.0; Stage3 fails all seeds
M4_sigma1.0,44,0.8991,18143.5,15742.5,20180.422,2401.0,0.4787,0.6828,0.6914,DFL sigma=1.0; Stage3 fails all seeds
M4_sigma2.0,42,0.9064,18292.5,15742.5,20180.422,2550.0,0.4681,0.6763,0.6842,DFL sigma=2.0; Stage3 improves seed43 only
M4_sigma2.0,43,0.8732,17622.5,15742.5,20180.422,1880.0,0.5106,0.7163,0.7550,DFL sigma=2.0; Stage3 improves seed43 only
M4_sigma2.0,44,0.8991,18143.5,15742.5,20180.422,2401.0,0.4787,0.6828,0.6914,DFL sigma=2.0; Stage3 improves seed43 only
M4_sigma4.0,42,0.8821,17801.5,15742.5,20180.422,2059.0,0.5000,0.7086,0.7229,DFL sigma=4.0; BEST
M4_sigma4.0,43,0.8820,17798.5,15742.5,20180.422,2056.0,0.4894,0.7117,0.7437,DFL sigma=4.0; BEST
M4_sigma4.0,44,0.8991,18143.5,15742.5,20180.422,2401.0,0.4787,0.6828,0.6914,DFL sigma=4.0; seed44 stuck
M4_sigma6.0,42,0.9064,18292.5,15742.5,20180.422,2550.0,0.4681,0.6763,0.6842,DFL sigma=6.0; Stage3 collapses all seeds
M4_sigma6.0,43,0.8871,17901.5,15742.5,20180.422,2159.0,0.4947,0.6928,0.7060,DFL sigma=6.0; Stage3 collapses all seeds
M4_sigma6.0,44,0.8991,18143.5,15742.5,20180.422,2401.0,0.4787,0.6828,0.6914,DFL sigma=6.0; Stage3 collapses all seeds
M4_sigma8.0,42,0.8812,17783.5,15742.5,20180.422,2041.0,0.5053,0.7116,0.7337,DFL sigma=8.0; partial recovery
M4_sigma8.0,43,0.8871,17901.5,15742.5,20180.422,2159.0,0.4947,0.6928,0.7060,DFL sigma=8.0; partial recovery
M4_sigma8.0,44,0.8991,18143.5,15742.5,20180.422,2401.0,0.4787,0.6828,0.6914,DFL sigma=8.0; seed44 stuck
```

---

## Aggregated Results (mean ± std across seeds 42/43/44)

| Method | C_norm ↓ | recall@K ↑ | pairwise_acc ↑ | AUC-ROC ↑ |
|--------|----------|------------|----------------|-----------|
| Oracle | 0.780 | — | — | — |
| Random | 1.000 | — | — | — |
| M1 | 0.9707 ± 0.0063 | 0.3688 ± 0.0067 | 0.5447 ± 0.0119 | 0.5283 ± 0.0155 |
| M2 | 0.8957 ± 0.0041 | 0.4770 ± 0.0050 | 0.6878 ± 0.0057 | 0.7039 ± 0.0099 |
| M3_old | 0.9043 ± 0.0104 | 0.4663 ± 0.0125 | 0.6795 ± 0.0112 | 0.6939 ± 0.0161 |
| M3_fix | 0.9004 ± 0.0081 | 0.4716 ± 0.0090 | 0.6818 ± 0.0093 | 0.6965 ± 0.0143 |
| M4 σ=0.5 | 0.8975 ± 0.0080 | 0.4805 ± 0.0109 | 0.6839 ± 0.0068 | 0.6939 ± 0.0091 |
| M4 σ=1.0 | 0.8975 ± 0.0080 | 0.4805 ± 0.0109 | 0.6839 ± 0.0068 | 0.6939 ± 0.0091 |
| M4 σ=2.0 | 0.8929 ± 0.0142 | 0.4858 ± 0.0181 | 0.6918 ± 0.0175 | 0.7102 ± 0.0318 |
| **M4 σ=4.0** | **0.8877 ± 0.0080** | **0.4894 ± 0.0087** | **0.7011 ± 0.0130** | **0.7193 ± 0.0215** |
| M4 σ=6.0 | 0.8975 ± 0.0080 | 0.4805 ± 0.0109 | 0.6839 ± 0.0068 | 0.6939 ± 0.0091 |
| M4 σ=8.0 | 0.8891 ± 0.0074 | 0.4929 ± 0.0109 | 0.6957 ± 0.0120 | 0.7104 ± 0.0176 |

**C_oracle = 15742.5, C_random = 20180.422 (fixed for all methods, test set N=333)**

---

## Key Findings

1. **M1→M2 gap is large** (~0.075 C_norm): adding a severity head with CE training gives most of the gain.

2. **M3 does not outperform M2**: checkpointing by val scheduling cost is noisier than val CE, even with 5-realization averaging. M3_fix reduces variance but doesn't close the gap.

3. **M4 DFL requires σ≈4.0 to work**: below σ=2.0, Stage 3 fails entirely (grad signal too small). Above σ=4.0, the signal degrades (perturbations scramble rankings). σ=4.0 is a sharp peak.

4. **M4 σ=4.0 beats M2** on all metrics: C_norm −0.008, recall@K +0.012, pairwise_acc +0.013, AUC-ROC +0.015.

5. **Seed 44 is stuck in Stage 3** for all σ: Stage 3 never improves over Stage 2 regardless of sigma. This appears to be a generalization failure (train surrogate decreases but val cost does not).

---

## Suggested Plots

- **Bar chart / box plot**: C_norm for M1, M2, M3_fix, M4_σ4.0 across seeds (main comparison)
- **Sigma sweep line plot**: mean C_norm vs σ (with ±std band) — shows the sharp peak at σ=4.0 and collapses at σ=0.5/1.0/6.0
- **Per-seed heatmap**: C_norm for all methods × seeds — highlights seed44 Stage 3 failure
- **Metric comparison radar/table**: C_norm, recall@K, pairwise_acc, AUC-ROC for M1/M2/M3/M4_best
