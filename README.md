# Glaucoma DFL Scheduling

> **Branch:** `main` — availability-constrained multi-slot scheduling (v4).
> For the unconstrained baseline see the `no-avail` branch.

---

## Project Overview

This project applies **Decision-Focused Learning (DFL)** to glaucoma referral scheduling. Rather than training a model to predict severity labels accurately and then handing those predictions to a scheduler, the DFL approach trains the model end-to-end with the downstream scheduling decision embedded in the learning objective.

Patients are assigned to one of T=3 appointment slots (or left unscheduled). The cost is delay-sensitive: more severe patients incur a higher penalty the longer they wait, and a large miss penalty if they are never scheduled.

**v4 upgrade (this branch):** Each patient has a binary availability vector indicating which slots they can attend. This transforms the problem from a pure ranking into a **constrained assignment** — the solver must respect who is available for each slot, making the assignment decision non-separable and more sensitive to prediction quality. This is the core setting where DFL is expected to have the largest advantage over two-stage methods.

---

## Model Design

Four methods are compared in an ablation study:

| Model | Description | Training Objective | Init |
|---|---|---|---|
| **M1** | Binary-only baseline | Binary CE on glaucoma/no-glaucoma labels | ImageNet |
| **M2a** | Severity CE, no warm-start | Multi-class CE on severity labels; val checkpoint = min decision cost | ImageNet |
| **M2b** | Severity CE + M1 warm-start | Same as M2a; initialised from M1 checkpoint (Cheap Thrills) | M1 |
| **M3** | DFL end-to-end | Stage 2 = M2b; Stage 3 = REINFORCE gradient through availability-constrained solver | M1 |

All four share the same **DualHeadVGG19** backbone (VGG19 + shared trunk + binary head + severity head). The triage score for M2/M3 is `α̂_i = Σ_k α_k · p_ik ∈ [0, 10]`, where `p_ik` is the predicted probability of severity grade k.

**M3 gradient estimation.** The scheduling solver is piecewise-constant, so gradients are zero almost everywhere. M3 smooths it via randomised perturbation (REINFORCE / score-function estimator):

```
ĝ ≈ (1/Mσ) Σ_m (C(z*(α̂ + σε_m), avail_batch) − baseline) · ε_m
```

A fresh availability matrix is sampled once per batch and shared across all M perturbations, so the gradient measures "what happens when scores change, holding constraints fixed."

---

## Repository Structure

```
glaucoma-dfl-scheduling/
├── config.py                       # shared hyperparameters (cost, LRs, epochs, seeds,
│                                   # availability probability and seeds)
├── pyproject.toml                  # project metadata and dependencies
│
├── data/
│   └── availability/               # fixed val/test .npy matrices (generated once before training)
│                                   # val_availability_seed100.npy, test_availability_seed200.npy
│
├── src/
│   ├── model.py                    # DualHeadVGG19 architecture
│   ├── dataset.py                  # GlaucomaDataset — loads manifest CSV + images
│   ├── allocation.py               # assign_slots(): unconstrained solver (training backward pass)
│   │                               # solve_multislot_availability(): constrained solver (eval + M2b/M3)
│   ├── losses.py                   # scheduling_cost_multislot() — differentiable cost function
│   ├── evaluate.py                 # framework-agnostic evaluation (CSV → metrics JSON)
│   │                               # oracle/random baselines also use constrained solver
│   ├── simulate_availability.py    # generate random (N, T) binary availability matrices
│   ├── generate_availability.py    # one-time script: save fixed val/test matrices to data/availability/
│   │
│   ├── train_M1.py                 # ← M1: binary CE training (no availability)
│   ├── train_M2a.py                # ← M2a: severity CE, ImageNet init, constrained val cost
│   ├── train_M2b.py                # ← M2b: severity CE, M1 warm-start, constrained val cost
│   ├── train_M3.py                 # ← M3: DFL fine-tuning with per-batch availability sampling
│   │
│   ├── dataset_construction/       # EyeAI catalog scripts (data pipeline)
│   │   ├── data_pipeline_v2.py     # downloads images and builds manifest CSV
│   │   └── construct_datasets_v2.py# creates train/val/test dataset splits in catalog
│   │
│   └── previous_exp/               # legacy Keras hyperparameters (best_hyperparameters.json)
│
├── tests/
│   └── test_solver_availability.py # unit tests for solve_multislot_availability()
│
└── docs/                           # design documents and experiment plans
    ├── v4_implementation_spec.md   # full v4 spec
    ├── experiment_plan_v4_availability.md  # research questions and experiment design
    ├── running_instructions.md     # step-by-step commands to run training and evaluation
    └── implementation_plan.md      # file-by-file porting notes
```

---

## Script → Model Mapping

| Script | Model | Key inputs | Key outputs |
|---|---|---|---|
| `src/train_M1.py` | M1 | manifest CSV, binary labels | `models/M1_seed{s}.pt`, `results/M1_seed{s}.csv` |
| `src/train_M2a.py` | M2a | manifest CSV, severity labels, `val_availability_seed100.npy` | `models/M2a_seed{s}.pt`, `results/M2a_seed{s}.csv` |
| `src/train_M2b.py` | M2b | manifest CSV, severity labels, `M1_seed{s}.pt`, val + test availability | `models/M2b_seed{s}.pt`, `results/M2b_seed{s}.csv`, `_metrics.json` |
| `src/train_M3.py` | M3 | manifest CSV, severity labels, `M1_seed{s}.pt`, val + test availability | `models/M3_seed{s}.pt`, `results/M3_seed{s}.csv`, `_metrics.json` |
| `src/evaluate.py` | all | prediction CSV, test availability `.npy` | metrics JSON (C_total, C_norm, C_regret, recall@K, AUC) |
| `src/generate_availability.py` | — | manifest CSV | `data/availability/val_availability_seed100.npy`, `test_availability_seed200.npy` |

Training order: **generate availability → M1 → M2a / M2b (parallel) → M3**.

---

## Quick Start

```bash
# 0. Generate fixed val/test availability matrices (once, before any training)
python src/generate_availability.py --manifest /data/lizhiwei/dfl_v2/manifest.csv

# 1. Evaluate existing M1 with availability-constrained solver
#    (rename old checkpoint first: M1_v3_seed42.pt → M1_seed42.pt)
python src/evaluate.py \
    --predictions /data/lizhiwei/dfl_v2/results/M1_v3_seed42.csv \
    --availability data/availability/test_availability_seed200.npy \
    --output /data/lizhiwei/dfl_v2/results/M1_seed42_metrics.json

# 2. Train M2b (requires M1 checkpoint + availability matrices)
python src/train_M2b.py --seed 42

# 3. Train M3 (requires M1 checkpoint + availability matrices)
python src/train_M3.py --seed 42
```

See `docs/running_instructions.md` for full instructions, multi-seed runs, and smoke tests.

---

## Cost Function

```
C(z, Y) = Σ_{i,t} z[i,t] · (α[y_i] · delay[t] + β)
         + Σ_i (1 − Σ_t z[i,t]) · α[y_i] · d_miss
```

with the added constraint `z[i,t] ≤ availability[i,t]` for all i, t.

Default parameters (from `config.py`): `α = [0,1,3,6,10]`, `delay = [1.0, 3.0, 8.0]`, `β = 0.5`, `d_miss = 15.0`, `K_frac_list = [0.05, 0.10, 0.20]`, `p_available = 0.7`.

Primary metric: `C_norm = C_total / C_random` (< 1 means better than random assignment). Both the random baseline and the oracle use the same availability-constrained solver so `C_norm` is comparable across methods.
