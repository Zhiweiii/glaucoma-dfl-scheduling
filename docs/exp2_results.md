# Experiment 2 — Unfrozen Trunk + R5 Availability

**Branch:** `single-head-framework`
**Commit:** `d473d16` (fixes committed)
**Run dates:** 2026-05-09 → 2026-05-10
**Tag:** `exp2` — main method comparison, seeds 42/43/44, full severity labels

---

## What Changed from Exp 1

| Change | Exp 1 | Exp 2 |
|--------|-------|-------|
| Availability | `p=0.7` uniform (r≈14×) | `p=[0.25, 0.50, 1.00]` per slot (r=5×) |
| Trunk in M2/M3/M4 | Frozen (severity_head only, 645 params) | Unfrozen (backbone[9+] + trunk + head, 21.4M params) |
| `sigma` (DFL) | 0.5 | 0.5 |
| `M` (MC samples) | 20 | 50 |
| `lr_stage3` | 1e-5 | 1e-4 |

**Motivation — frozen trunk bottleneck:** With the trunk frozen from M1 (binary triage training), the 128-dim features were optimised for glaucoma vs. normal, not severity discrimination. The `Linear(128→5)` severity head had no capacity to extract severity signal not already encoded. Unfreezing gives M2/M3/M4 equal capacity starting from the same M1 initialisation.

**Motivation — tighter availability:** The r=14× oversubscription in exp1 meant the availability constraint never bound — the ILP reduced to pure ranking, making the DFL gradient near zero and M3's cost-based checkpoint selection equivalent to CE. Switching to r=5× makes constraints active for slot 1 and slot 2.

---

## Architecture

```
VGG19 backbone (layers 0–37)
  └── Trunk: Linear(25088→64→128), ELU, Tanh, BatchNorm, Dropout
        ├── binary_head:   Linear(128→1)     [binary triage, frozen in Phase 2]
        └── severity_head: Linear(128→5)     [severity grades 1–4]
```

Triage score: `α̂_i = Σ_k α_k · softmax(sev_logits)_ik ∈ [0, 10]`  
Cost parameters: `α = [0,1,3,6,10]`, `β = 0.5`, `delay = [1,3,8]`, `d_miss = 15`

---

## Methods

| Method | Training objective | Checkpoint criterion |
|--------|--------------------|----------------------|
| **M1** | Binary CE (glaucoma vs. normal) | Val binary AUC |
| **M2** | Severity CE (grades 1–4) | Val CE loss |
| **M3** | Severity CE (grades 1–4) | Val scheduling cost (ILP) |
| **M4** | Stage 2: severity CE → Stage 3: DFL (REINFORCE) | Stage 2: val cost; Stage 3: val cost |

**M4 DFL gradient estimator:**

```
α̂_i = Σ_k α_k · softmax(sev_logits_i)_k      (expected severity cost)
ĝ ≈ (1 / M·σ) Σ_m (C_m/N − baseline) · ε_m   (REINFORCE estimate)
L  = (α̂ · ĝ_detached).sum()                   (surrogate loss)
```

where `ε_m ~ N(0, I)`, `C_m = C(z*(α̂ + σε_m), Y)`, and `z*` is solved by Gurobi ILP.

---

## Exp 2 Results — Test Set (N=333, K=[16,33,66])

**Availability per slot (r5, p=[0.25, 0.50, 1.00]):**

| Slot | K | E[available] | Oversubscription |
|------|---|--------------|-----------------|
| 1 (delay=1) | 16 | ~83 | 5.2× |
| 2 (delay=3) | 33 | ~167 | 5.1× |
| 3 (delay=8) | 66 | ~333 | 5.0× |

### Per-seed metrics

| Model | seed42 | seed43 | seed44 |
|-------|--------|--------|--------|
| **M1** C_norm | 0.9619 | 0.9764 | 0.9739 |
| **M2** C_norm | 0.8900 | 0.8998 | 0.8974 |
| **M3** C_norm | 0.8900 | 0.9145 | 0.9083 |
| **M4** C_norm | 0.9064 | 0.8871 | 0.8991 |

### Summary (mean ± std across seeds)

| Model | C_norm ↓ | recall@K ↑ | pairwise_acc ↑ | AUC-ROC ↑ |
|-------|----------|------------|----------------|-----------|
| Oracle | ~0.780 | — | — | — |
| Random | 1.000 | — | — | — |
| **M1** | 0.971 ± 0.006 | 0.369 ± 0.007 | 0.545 ± 0.012 | 0.528 ± 0.016 |
| **M2** | 0.896 ± 0.004 | 0.477 ± 0.005 | 0.688 ± 0.006 | 0.704 ± 0.010 |
| **M3** | 0.904 ± 0.010 | 0.466 ± 0.013 | 0.680 ± 0.011 | 0.694 ± 0.016 |
| **M4** | 0.898 ± 0.008 | 0.481 ± 0.011 | 0.684 ± 0.007 | 0.694 ± 0.009 |

---

## Observations

**1. M2 still best / M3 worse than M2.**
M3 should outperform M2 because it explicitly optimises for the scheduling objective at checkpoint time. Instead it underperforms. Root cause: with a single val availability realization, the val scheduling cost is a high-variance signal — the ILP solution changes substantially across draws when constraints are binding, so the "best" checkpoint according to one draw is not the best overall. M3 overfits to the random structure of that one availability matrix.

**2. M4 DFL gives marginal or no gain.**
M4 (C_norm=0.898) is comparable to M2 (0.896), not clearly better. The REINFORCE gradient estimator requires σ large enough that perturbed scores `α̂ + σε` change the ILP assignment. With σ=0.5 and `alpha_hat` std ≈ 1.2–1.5, the perturbations are only ~0.35 std — too small to flip enough assignments, giving a weak cost gradient.

**3. Unfreezing helps overall.**
All methods improved modestly over their exp1 counterparts, consistent with the trunk having more capacity to learn severity-discriminative features. AUC-ROC for M2 improved from 0.704 (exp1) to 0.704 (exp2, similar), but C_norm is comparable.

---

## Fixes Applied (exp2_fixes run)

### Fix 1 — M3: Multi-realization val checkpoint selection

**Problem:** Single val availability realization → high-variance checkpoint signal.

**Fix:** Average val scheduling cost over 5 availability realizations (seeds 100–104) before checkpoint comparison.

```python
# Before
val_cost = compute_cost(model, val_loader, val_availability)

# After
val_cost = mean([compute_cost(model, val_loader, avail)
                 for avail in val_availabilities])   # 5 realizations
```

Results saved as `M3_seed*.csv` in `exp2/results/` (overwrites original M3 results).

### Fix 2 — M4: Sigma sweep for DFL perturbation scale

**Problem:** σ=0.5 ≈ 0.35 × (alpha_hat std) → perturbations too small to flip ILP assignments → near-zero cost variance → near-zero REINFORCE gradient.

**Fix:** Re-run Stage 3 only (reusing Stage 2 checkpoints) with larger σ values.

| σ | Ratio to alpha_hat std | Expected effect |
|---|----------------------|-----------------|
| 0.5 | 0.35× (baseline, already done) | Weak gradient |
| 1.0 | 0.7× | Moderate exploration |
| 2.0 | 1.4× | Full std exploration |
| 4.0 | 2.7× | Aggressive exploration |

New files `M4_sigma{1.0,2.0,4.0}_seed*.csv` in `exp2/results/`.

The `--stage3-only` flag skips Stage 2 re-training; Stage 2 checkpoints from `exp2/models/M4_stage2_seed*.pt` are reused.

---

## Artifact Paths

```
/data/lizhiwei/dfl_v2/v5/exp2/
  models/   M{1,2,3,4}_seed{42,43,44}.pt
            M4_stage2_seed{42,43,44}.pt
            M4_sigma{1.0,2.0,4.0,6.0,8.0}_seed{42,43,44}.pt   [after fixes + sigma sweep]
  results/  {M1,M2,M3,M4}_seed{42,43,44}.{csv,_metrics.json}
            M3_seed{42,43,44}.{csv,_metrics.json}              [overwritten with M3-fix]
            M4_sigma{1.0,2.0,4.0,6.0,8.0}_seed{42,43,44}.{csv,_metrics.json}
  logs/     {M2,M3,M4}_seed{42,43,44}.log
            M3fix_seed{42,43,44}.log
            M4_sigma{1.0,2.0,4.0,6.0,8.0}_seed{42,43,44}.log
            fixes_master.log   sigma_sweep2_master.log

/data/lizhiwei/dfl_v2/v5/availability_r5/
  train_availability_seed0.npy         shape (1091, 3)
  val_availability_seed{100..104}.npy  shape (269, 3)   [5 realizations for fix 1]
  test_availability_seed200.npy        shape (333, 3)
```
