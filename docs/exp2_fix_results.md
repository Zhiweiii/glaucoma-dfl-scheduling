# Exp 2 Fix Results — M3 Checkpoint Variance Fix and M4 Sigma Sweep

**Branch:** `single-head-framework`
**Commit:** `1c8b9ba` (run_exp2_fixes.sh)
**Run date:** 2026-05-10
**Script:** `scripts/run_exp2_fixes.sh`

---

## Fix 1 — M3: Multi-Realization Checkpoint Selection

**Change:** Val scheduling cost averaged over 5 availability realizations (seeds 100–104)
instead of 1, to reduce the noise in checkpoint selection when constraints are binding.

### Per-seed results

| | seed42 | seed43 | seed44 | Mean ± Std |
|---|--------|--------|--------|------------|
| **M3 (old)** C_norm | 0.8900 | 0.9145 | 0.9083 | 0.9043 ± 0.0104 |
| **M3 (fix)** C_norm | 0.8900 | 0.9098 | 0.9015 | 0.9004 ± 0.0081 |
| **M3 (fix)** recall@K | 0.4840 | 0.4628 | 0.4681 | 0.4716 ± 0.0090 |
| **M3 (fix)** pairwise | 0.6949 | 0.6753 | 0.6750 | 0.6818 ± 0.0093 |
| **M3 (fix)** AUC-ROC | 0.7166 | 0.6870 | 0.6858 | 0.6965 ± 0.0143 |

**Outcome:** Modest improvement — mean C_norm 0.9043 → 0.9004, variance reduced (±0.0104 → ±0.0081).
M3 still underperforms M2 (0.8957). The multi-realization fix reduces noise but does not close the gap,
suggesting the CE loss is a more reliable training objective than val cost for checkpoint selection
when the cost signal is still relatively noisy.

---

## Fix 2 — M4 Stage 3: Sigma Sweep

**Change:** Stage 3 (DFL) re-run with larger perturbation σ using existing Stage 2 checkpoints
(`--stage3-only`). σ=0.5 (default) was already run in exp2; new values: 1.0, 2.0, 4.0.

**Rationale:** With alpha_hat std ≈ 1.5, σ=0.5 is only 0.33× the natural score spread — too small
to reliably flip ILP assignments across MC samples, giving near-zero cost variance in the
REINFORCE estimator.

### Per-seed C_norm results

| Method | seed42 | seed43 | seed44 | Mean ± Std |
|--------|--------|--------|--------|------------|
| M2 (CE baseline) | 0.8900 | 0.8998 | 0.8974 | 0.8957 ± 0.0041 |
| M4 σ=0.5 | 0.9064 | 0.8871 | 0.8991 | 0.8975 ± 0.0080 |
| M4 σ=1.0 | 0.9064 | 0.8871 | 0.8991 | 0.8975 ± 0.0080 |
| M4 σ=2.0 | 0.9064 | **0.8732** | 0.8991 | 0.8929 ± 0.0142 |
| **M4 σ=4.0** | **0.8821** | **0.8820** | 0.8991 | **0.8877 ± 0.0080** |

### Full metrics at σ=4.0 (best)

| | seed42 | seed43 | seed44 | Mean ± Std |
|---|--------|--------|--------|------------|
| C_norm ↓ | 0.8821 | 0.8820 | 0.8991 | **0.8877 ± 0.0080** |
| recall@K ↑ | 0.5000 | 0.4894 | 0.4787 | **0.4894 ± 0.0087** |
| pairwise ↑ | 0.7086 | 0.7117 | 0.6828 | **0.7011 ± 0.0130** |
| AUC-ROC ↑ | 0.7229 | 0.7437 | 0.6914 | **0.7193 ± 0.0215** |

### Comparison summary

| Method | C_norm ↓ | recall@K ↑ | pairwise ↑ | AUC-ROC ↑ |
|--------|----------|------------|------------|-----------|
| M2     | 0.8957 ± 0.0041 | 0.4770 ± 0.0050 | 0.6878 ± 0.0057 | 0.7039 ± 0.0099 |
| M3 (fix) | 0.9004 ± 0.0081 | 0.4716 ± 0.0090 | 0.6818 ± 0.0093 | 0.6965 ± 0.0143 |
| M4 σ=4.0 | **0.8877 ± 0.0080** | **0.4894 ± 0.0087** | **0.7011 ± 0.0130** | **0.7193 ± 0.0215** |

**M4 σ=4.0 improves over M2 on all metrics.** C_norm reduces by 0.0080 (roughly 2× M2's std),
recall@K gains +0.012, pairwise accuracy gains +0.013.

---

## Stage 3 Behaviour by Seed

| Seed | Stage 2 val_cost | Stage 3 improved? | Best σ |
|------|-----------------|-------------------|--------|
| 42 | ~12800 | Yes (σ≥4.0) | σ=4.0 |
| 43 | ~12400 | Yes (σ≥2.0) | σ=2.0 or 4.0 |
| 44 | 12553 | **Never** (all σ) | — |

Seed 44 Stage 3 is stuck across all sigma values. At σ=4.0 the gradient norm is
healthy (≈0.5) and train surrogate decreases, but the improvement does not transfer to val.
This is a generalization issue: the model learns to exploit the fixed training availability
structure rather than learning a more general severity ranking. Seed 44's Stage 2 model may
be at a saddle point in the DFL objective from which Stage 3 cannot escape in 20 epochs.

---

## Should We Sweep Larger σ?

**Short answer: one more point (σ=8.0) is worth trying, but we are likely near the plateau.**

The trend from σ=0.5 → 1.0 → 2.0 → 4.0 shows:
- σ=0.5 and σ=1.0 are **identical** — Stage 3 fails completely, falling back to Stage 2 for all seeds.
  The threshold for DFL to work at all is somewhere between σ=1.0 and σ=2.0.
- σ=2.0: Stage 3 improves for seed 43 only.
- σ=4.0: Stage 3 improves for seeds 42 and 43; seed 44 remains stuck.

**Why σ=8.0 may not help further:**
- alpha_hat ∈ [0, 10] by construction. At σ=8, a perturbation of ±8 scrambles
  the ranking almost entirely — the ILP solutions for different ε samples become
  nearly independent of the true scores, so `C_m` variance collapses back toward zero
  and the gradient signal degrades again (same failure mode as σ too small, but from
  the other direction).
- Seed 44's failure is a generalization problem, not a signal problem. Larger σ won't fix it.
- Seeds 42 and 43 already improved substantially at σ=4.0; further gains would be small.

**Recommendation:** Run σ=8.0 as a single confirmation point. If it is worse than σ=4.0,
we have found the peak and can report σ=4.0 as the best configuration. If it is better,
try σ=6.0 to bracket the optimum. Either way, expect diminishing returns and seed 44
to remain unimproved.

---

## Artifact Paths

```
/data/lizhiwei/dfl_v2/v5/exp2/results/
  M3_seed{42,43,44}.{csv,_metrics.json}              # M3-fix results (overwrote old M3)
  M4_sigma{1.0,2.0,4.0}_seed{42,43,44}.{csv,_metrics.json}

/data/lizhiwei/dfl_v2/v5/exp2/models/
  M3_seed{42,43,44}.pt                               # M3-fix checkpoints
  M4_sigma{1.0,2.0,4.0}_seed{42,43,44}.pt

/data/lizhiwei/dfl_v2/v5/exp2/logs/
  fixes_master.log
  M3fix_seed{42,43,44}.log
  M4_sigma{1.0,2.0,4.0}_seed{42,43,44}.log
```
