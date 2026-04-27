# Glaucoma DFL Scheduling — Unconstrained Baseline (`no-avail`)

> **Branch:** `no-avail` — multi-slot scheduling without availability constraints (v3).
> For the availability-constrained version see the `main` branch.

---

## Project Overview

This project applies **Decision-Focused Learning (DFL)** to glaucoma referral scheduling. Rather than training a model to predict severity labels accurately and then handing those predictions to a scheduler, the DFL approach trains the model end-to-end with the downstream scheduling decision embedded in the learning objective.

Patients are assigned to one of T=3 appointment slots (or left unscheduled). The cost is delay-sensitive: more severe patients incur a higher penalty the longer they wait, and a large miss penalty if they are never scheduled. The model learns to produce triage scores that minimise this cost directly.

This branch (`no-avail`) is the **unconstrained baseline**: every patient is eligible for any slot. The `main` branch upgrades the problem with per-patient availability constraints, where each patient can only attend a subset of slots.

---

## Model Design

Four methods are compared in an ablation study:

| Model | Description | Training Objective | Init |
|---|---|---|---|
| **M1** | Binary-only baseline | Binary CE on glaucoma/no-glaucoma labels | ImageNet |
| **M2a** | Severity CE, no warm-start | Multi-class CE on severity labels; val checkpoint = min decision cost | ImageNet |
| **M2b** | Severity CE + M1 warm-start | Same as M2a; initialised from M1 checkpoint | M1 |
| **M3** | DFL end-to-end | Stage 2 = M2b; Stage 3 = REINFORCE gradient through scheduling solver | M1 |

All four share the same **DualHeadVGG19** backbone (VGG19 + shared trunk + binary head + severity head). The triage score for M2/M3 is `α̂_i = Σ_k α_k · p_ik ∈ [0, 10]`, where `p_ik` is the predicted probability of severity grade k.

**M3 gradient estimation.** The scheduling solver is piecewise-constant, so gradients are zero almost everywhere. M3 smooths it via randomised perturbation:

```
ĝ ≈ (1/Mσ) Σ_m (C(z*(α̂ + σε_m)) − baseline) · ε_m
```

The surrogate loss `L = (α̂ · ĝ_detached).sum()` has `∂L/∂α̂ = ĝ`, which propagates back through the network.

---

## Repository Structure

```
glaucoma-dfl-scheduling/
├── config.py                       # shared hyperparameters (cost, LRs, epochs, seeds)
├── pyproject.toml                  # project metadata and dependencies
│
├── data/
│   └── availability/               # (empty on this branch — no availability matrices)
│
├── src/
│   ├── model.py                    # DualHeadVGG19 architecture
│   ├── dataset.py                  # GlaucomaDataset — loads manifest CSV + images
│   ├── allocation.py               # assign_slots(): unconstrained greedy multi-slot solver
│   ├── losses.py                   # scheduling_cost_multislot() — differentiable cost
│   ├── evaluate.py                 # framework-agnostic evaluation (CSV → metrics JSON)
│   │
│   ├── train_M1.py                 # ← M1: binary CE training
│   ├── train_M2a.py                # ← M2a: severity CE, ImageNet init
│   ├── train_M2b.py                # ← M2b: severity CE, M1 warm-start
│   ├── train_M3.py                 # ← M3: DFL fine-tuning (Stage 2 + Stage 3)
│   │
│   ├── dataset_construction/       # EyeAI catalog scripts (data pipeline)
│   │   ├── data_pipeline_v2.py     # downloads images and builds manifest CSV
│   │   └── construct_datasets_v2.py# creates train/val/test dataset splits in catalog
│   │
│   └── previous_exp/               # legacy Keras hyperparameters (best_hyperparameters.json)
│
├── tests/                          # unit tests
└── docs/                           # design documents and experiment plans
```

---

## Script → Model Mapping

| Script | Model | Key inputs | Key outputs |
|---|---|---|---|
| `src/train_M1.py` | M1 | manifest CSV, binary labels | `models/M1_v3_seed{s}.pt`, `results/M1_v3_seed{s}.csv` |
| `src/train_M2a.py` | M2a | manifest CSV, severity labels | `models/M2a_seed{s}.pt`, `results/M2a_seed{s}.csv` |
| `src/train_M2b.py` | M2b | manifest CSV, severity labels, `M1_v3_seed{s}.pt` | `models/M2b_seed{s}.pt`, `results/M2b_seed{s}.csv` |
| `src/train_M3.py` | M3 | manifest CSV, severity labels, `M1_v3_seed{s}.pt` | `models/M3_seed{s}.pt`, `results/M3_seed{s}.csv` |
| `src/evaluate.py` | all | prediction CSV | metrics JSON (C_total, C_norm, C_regret, recall@K, AUC) |

Training order: **M1 → M2a / M2b (parallel) → M3**.

---

## Quick Start

```bash
# 1. Train M1 (produces the checkpoint that M2b and M3 need)
python src/train_M1.py --seed 42

# 2. Train M2b (requires M1 checkpoint)
python src/train_M2b.py --seed 42

# 3. Train M3 (requires M1 checkpoint; re-runs Stage 2 then adds Stage 3 DFL)
python src/train_M3.py --seed 42

# 4. Evaluate any model
python src/evaluate.py \
    --predictions /data/lizhiwei/dfl_v2/results/M1_v3_seed42.csv \
    --output /data/lizhiwei/dfl_v2/results/M1_v3_seed42_metrics.json
```

See `docs/running_instructions.md` for full instructions including multi-seed runs and smoke tests.

---

## Cost Function

```
C(z, Y) = Σ_{i,t} z[i,t] · (α[y_i] · delay[t] + β)
         + Σ_i (1 − Σ_t z[i,t]) · α[y_i] · d_miss
```

Default parameters (from `config.py`): `α = [0,1,3,6,10]`, `delay = [1.0, 3.0, 8.0]`, `β = 0.5`, `d_miss = 15.0`, `K_frac_list = [0.05, 0.10, 0.20]`.

Primary metric: `C_norm = C_total / C_random` (< 1 means better than random assignment).
