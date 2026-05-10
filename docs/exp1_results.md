# Experiment 1 — V5 Single-Head Framework (Baseline)

**Branch:** `single-head-framework`
**Commit:** `5bf9ad27a5c54276e3a65d04b51ea01463ba15b5`
**Run dates:** 2026-05-06 → 2026-05-09
**Tag:** `exp1` — main method comparison, seeds 42/43/44, full severity labels

---

## Design

Four methods share the same VGG19 backbone and two-phase training structure:

| Method | Description |
|--------|-------------|
| **M1** | Binary triage only (no severity head). Triage score = glaucoma probability. |
| **M2** | Severity CE fine-tune on top of M1. Checkpoint by **val CE loss**. |
| **M3** | Severity CE fine-tune on top of M1. Checkpoint by **val scheduling cost** (light-touch). |
| **M4** | M3 Stage 2 (CE) + Stage 3 DFL fine-tuning via score-function gradient estimator (REINFORCE). |

All methods use the same architecture: VGG19 backbone → shared trunk → severity head.  
Triage score: `α̂_i = Σ_k α_k · softmax(sev_logits)_ik ∈ [0, 10]`.

---

## Hyperparameters / Config

### Cost function
| Param | Value | Notes |
|-------|-------|-------|
| `alpha` | `[0, 1, 3, 6, 10]` | Severity costs for grades 0–4 |
| `beta` | `0.5` | Per-referral fixed cost |
| `delay` | `[1.0, 3.0, 8.0]` | Delay weights for slots 1–3 |
| `d_miss` | `15.0` | Miss penalty multiplier |
| `K_frac_list` | `[0.05, 0.10, 0.20]` | Slot capacities as fraction of N |

**Capacity in absolute terms (test N=333):** K = [16, 33, 66], total = 115/333 = **34.5% scheduled**.

### Availability
| Param | Value |
|-------|-------|
| `p_available` | `0.7` (uniform, per slot) |
| `T` | 3 slots |
| `seed_train / val / test` | `0 / 100 / 200` |

**Availability vs capacity (test set):**

| Slot | K | Expected available @p=0.7 | Oversubscribed |
|------|---|--------------------------|----------------|
| 1 (delay=1) | 16 | 233 | **14.6×** |
| 2 (delay=3) | 33 | 233 | **7.1×** |
| 3 (delay=8) | 66 | 233 | **3.5×** |

> **Known issue:** `p_available=0.7` is far too generous relative to K_frac. The availability
> constraint never binds — the solver degenerates to pure score ranking. This explains
> why DFL (M4) adds no value over CE (M2/M3).

### Training
| Param | Value |
|-------|-------|
| `backbone` | VGG19 |
| `img_size` | 224 |
| `batch_size` | 32 |
| `lr_head` | `1e-4` (trunk + severity head) |
| `lr_finetune` | `~8.89e-7` (backbone layers 9+) |
| `epochs_stage2` | 30 |
| `patience` | 10 |
| `use_class_weights` | True |

### M4 Stage 3 (DFL)
| Param | Value |
|-------|-------|
| `lr_stage3` | `1e-4` |
| `epochs_stage3` | 20 |
| `sigma` | `0.5` |
| `M` (MC samples) | 50 |
| `batch_size_stage3` | 256 |

---

## Results — Test Set (N=333, K=[16,33,66])

Oracle C_norm = 15714.5 / 20157.141 = **0.7797**  
Random C_norm = **1.0000** (by definition)

### Per-seed C_norm (lower = better)

| Model | seed42 | seed43 | seed44 | Mean ± Std |
|-------|--------|--------|--------|------------|
| M1 | 0.9691 | 0.9665 | 0.9852 | 0.974 ± 0.009 |
| M2 | 0.8740 | 0.8865 | 0.8886 | 0.883 ± 0.006 |
| M3 | **0.8723** | 0.8967 | 0.8934 | 0.887 ± 0.011 |
| M4 | 0.8924 | **0.8811** | 0.8916 | 0.888 ± 0.005 |

### Per-seed recall@K

| Model | seed42 | seed43 | seed44 | Mean ± Std |
|-------|--------|--------|--------|------------|
| M1 | 0.367 | 0.378 | 0.340 | 0.362 ± 0.016 |
| M2 | **0.495** | 0.468 | 0.473 | 0.479 ± 0.012 |
| M3 | **0.500** | 0.457 | 0.468 | 0.475 ± 0.018 |
| M4 | 0.463 | 0.473 | **0.479** | 0.472 ± 0.007 |

### Per-seed pairwise accuracy

| Model | seed42 | seed43 | seed44 | Mean ± Std |
|-------|--------|--------|--------|------------|
| M1 | 0.561 | 0.538 | 0.534 | 0.545 ± 0.012 |
| M2 | 0.695 | 0.687 | 0.681 | 0.688 ± 0.006 |
| M3 | **0.702** | 0.675 | 0.675 | 0.684 ± 0.013 |
| M4 | 0.676 | **0.693** | 0.680 | 0.683 ± 0.007 |

### Per-seed AUC-ROC

| Model | seed42 | seed43 | seed44 | Mean ± Std |
|-------|--------|--------|--------|------------|
| M1 | 0.548 | 0.527 | 0.510 | 0.528 ± 0.016 |
| M2 | **0.717** | 0.703 | 0.692 | 0.704 ± 0.010 |
| M3 | **0.716** | 0.687 | 0.686 | 0.696 ± 0.014 |
| M4 | 0.684 | 0.706 | 0.689 | 0.693 ± 0.009 |

### Summary (mean ± std across seeds)

| Model | C_norm ↓ | recall@K ↑ | pairwise_acc ↑ | AUC-ROC ↑ |
|-------|----------|------------|----------------|-----------|
| Oracle | 0.780 | — | — | — |
| Random | 1.000 | — | — | — |
| **M1** | 0.974 ± 0.009 | 0.362 ± 0.016 | 0.545 ± 0.012 | 0.528 ± 0.016 |
| **M2** | 0.883 ± 0.006 | 0.479 ± 0.012 | 0.688 ± 0.006 | 0.704 ± 0.010 |
| **M3** | 0.887 ± 0.011 | 0.475 ± 0.018 | 0.684 ± 0.013 | 0.696 ± 0.014 |
| **M4** | 0.888 ± 0.005 | 0.472 ± 0.007 | 0.683 ± 0.007 | 0.693 ± 0.009 |

---

## Key Observations

1. **M2 ≈ M3 ≈ M4**: All three are statistically indistinguishable. The light-touch checkpoint
   selection (M3) and DFL fine-tuning (M4) add negligible improvement over plain CE (M2).

2. **M4 Stage 3 DFL fails in all seeds**: val_cost increases immediately on epoch 0 and never
   recovers. Stage 3 was not used for any final M4 checkpoint. Root cause: REINFORCE gradient
   estimator has C_m std/mean ≈ 0.6% (nearly zero signal), because with p_available=0.7 the
   availability constraints never bind and the cost landscape is nearly flat w.r.t.
   score perturbations (σ=0.5).

3. **M1 is clearly weaker**: adding a severity head provides substantial gains (~0.09 C_norm).

4. **M2 lowest variance**: the most stable method across seeds.

---

## Artifact Paths

All artifacts under `/data/lizhiwei/dfl_v2/v5/exp1/`

### Models
```
/data/lizhiwei/dfl_v2/v5/exp1/models/
  M1_seed{42,43,44}.pt                  # M1 final checkpoints
  M2_seed{42,43,44}.pt                  # M2 final checkpoints
  M3_seed{42,43,44}.pt                  # M3 final checkpoints
  M4_seed{42,43,44}.pt                  # M4 final checkpoints (= Stage 2 best; Stage 3 did not improve)
  M4_stage2_seed{42,43,44}.pt           # M4 Stage 2 best (same as M4 final)
```

### Results
```
/data/lizhiwei/dfl_v2/v5/exp1/results/
  {M1,M2,M3,M4}_seed{42,43,44}.csv           # test predictions (patient_id, triage_score, true_severity)
  {M1,M2,M3,M4}_seed{42,43,44}_metrics.json  # scheduling metrics
```

### Logs
```
/data/lizhiwei/dfl_v2/v5/exp1/logs/
  M1_seed{42,43,44}.log
  M2_seed{42,43,44}.log
  M3_seed{42,43,44}.log
  M4_seed{42,43,44}.log
```

### Availability matrices
```
/data/lizhiwei/dfl_v2/v5/availability/
  train_availability_seed0.npy    # shape (1091, 3)
  val_availability_seed100.npy    # shape (269, 3)
  test_availability_seed200.npy   # shape (333, 3)
```

### Source code (git)
```
Branch: single-head-framework
Commit: 5bf9ad27a5c54276e3a65d04b51ea01463ba15b5
  src/train_M1.py
  src/train_M2.py
  src/train_M3.py
  src/train_M4.py
  src/allocation.py       # Gurobi ILP solver
  src/losses.py           # scheduling_cost_multislot
  src/dataset.py
  src/generate_availability.py
  config.py
```

---

## Identified Issues for Next Experiment

1. **`p_available=0.7` is too high**: availability never constrains the solution. Need per-slot
   or much lower uniform probability so that capacity is the binding constraint. Suggested:
   per-slot `p = [0.10, 0.20, 0.40]` (≈2× oversubscription per slot), or uniform `p ≈ 0.40`.

2. **DFL gradient signal is near zero**: consequence of (1). Fixing availability will make the
   cost landscape non-flat and give REINFORCE real signal to exploit.

3. **M3 light-touch marginal benefit**: when CE and scheduling cost correlate well (good AUC),
   switching the checkpoint criterion gives little gain. May become more informative once
   availability constraints bind.
