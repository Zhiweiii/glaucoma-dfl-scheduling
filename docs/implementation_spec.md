# V5 Implementation Spec: Feature-Freezing Decomposition

This document specifies exactly what to implement for v5. It assumes familiarity with the v4 codebase.

---

## Model Naming

| Model | Role | Training objective | Checkpoint selection |
|---|---|---|---|
| M1 | Binary classifier — Phase 1 feature extractor | BCE on binary labels | Val BCE |
| M2 | Baseline severity head | CE on severity labels (grades 1–4) | Val CE loss |
| M3 | Light-touch severity head | CE on severity labels (grades 1–4) | Val scheduling cost |
| M4 | Full DFL severity head | Scheduling cost via perturbation gradients | Val scheduling cost |

---

## Summary of Changes from V4

The core change: **freeze the backbone and trunk during Phase 2 training for all severity models (M2, M3, M4).** Grade-0 is excluded from Phase 2 entirely — training, validation, and evaluation.

Most of the v4 infrastructure (availability constraints, solver, evaluation pipeline, cost function) is unchanged. The changes are concentrated in the training scripts and the data flow.

---

## What Does NOT Change

| Component | Status |
|---|---|
| `src/model.py` (DualHeadVGG19) | Unchanged — architecture stays the same |
| `src/losses.py` (scheduling_cost_multislot) | Unchanged |
| `src/allocation.py` (assign_slots, solve_multislot_availability) | Unchanged |
| `src/simulate_availability.py` | Unchanged |
| `tests/test_solver_availability.py` | Unchanged |
| `config.py` — scheduling parameters (K_frac_list, delay, d_miss, alpha) | Unchanged |
| `config.py` — availability parameters (p_available, seeds) | Unchanged |
| `config.py` — DFL parameters (sigma, M, batch_size_stage3) | Unchanged |

---

## What Changes

### 1. `src/train_M1.py` — NO CHANGES

M1 trains the full network (backbone + trunk + binary head) on binary labels. This is Phase 1.

Its output checkpoint provides the frozen feature extractor for all Phase 2 models.

**Output:** `M1_seed{seed}.pt` — starting point for M2, M3, and M4.

---

### 2. `src/train_M2a.py` → `src/train_M2.py` — RENAME + MODIFY

**Current behavior (v4):** Two-phase training where Phase 1 freezes backbone but trains trunk + severity head, then Phase 2 unfreezes backbone from layer 9.

**New behavior (v5):** Single-phase training. Backbone and trunk are permanently frozen. Only the severity head is trainable.

```python
# Load M1 checkpoint
model = DualHeadVGG19(...)
model.load_state_dict(torch.load(m1_checkpoint_path))

# Freeze everything except severity_head
for name, param in model.named_parameters():
    if "severity_head" not in name:
        param.requires_grad = False

# Single optimizer — only severity_head params
optimizer = torch.optim.Adam(
    model.severity_head.parameters(),
    lr=CONFIG["lr_head"]
)
```

**Training data:** Grades 1–4 only (`exclude_grade0=True` on all splits).

**Checkpoint selection:** Validation CE loss. Early stopping on validation CE.

**Output:** `M2_seed{seed}.pt`, `M2_seed{seed}.csv`, `M2_seed{seed}_metrics.json`

---

### 3. `src/train_M2b.py` → `src/train_M3.py` — RENAME + MODIFY

**New behavior (v5):** Identical to M2 except for checkpoint selection criterion.

```python
# Same freezing as M2
for name, param in model.named_parameters():
    if "severity_head" not in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(
    model.severity_head.parameters(),
    lr=CONFIG["lr_head"]
)
```

**Training loss:** Cross-entropy on severity labels (grades 1–4).

**Checkpoint selection (the light-touch part):** Validation scheduling cost, NOT validation CE. This is the only difference from M2.

```python
# After each epoch:
z_val = solve_multislot_availability(val_scores, K_list, val_availability)
val_cost = scheduling_cost_multislot(z_val, Y_val, ...)

if val_cost < best_val_cost:
    best_val_cost = val_cost
    save_checkpoint(model, epoch)
```

**Training data:** Grades 1–4 only (`exclude_grade0=True` on all splits).

**Output:** `M3_seed{seed}.pt`, `M3_seed{seed}.csv`, `M3_seed{seed}_metrics.json`

---

### 4. `src/train_M3.py` → `src/train_M4.py` — RENAME + MODIFY

**New behavior (v5):** Two stages — Stage 2 (CE training, identical to M3), then Stage 3 (DFL fine-tuning of severity head only).

```python
# ---- Stage 2: identical to M3 ----
# Load M1 checkpoint, freeze backbone + trunk, train severity_head with CE
# Checkpoint selection by validation scheduling cost
# Save as M4_stage2_seed{seed}.pt

# ---- Stage 3: DFL fine-tuning ----
# Load Stage 2 best checkpoint
# Backbone and trunk remain frozen — only severity_head is trainable
for name, param in model.named_parameters():
    if "severity_head" not in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(
    model.severity_head.parameters(),
    lr=CONFIG["lr_stage3"]
)
```

**DFL training loop:** Same perturbation gradient mechanism as v4. Gradients only flow through the severity head — `backward()` naturally stops at frozen layers. No structural change to `dfl_step` needed.

**Training data:** Grades 1–4 only (`exclude_grade0=True` on all splits including `dfl_loader`).

**Availability in DFL:** Same as v4 — fixed per-patient `train_availability`, indexed by `patient_idx`.

**Checkpoint selection:** Validation scheduling cost (same as M3).

**Output:** `M4_stage2_seed{seed}.pt`, `M4_seed{seed}.pt`, `M4_seed{seed}.csv`, `M4_seed{seed}_metrics.json`

---

### 5. `src/dataset.py` — MODIFY

Add a filtering option to exclude grade-0 from severity splits.

```python
class GlaucomaDataset:
    def __init__(self, ..., exclude_grade0=False):
        ...
        if exclude_grade0:
            self.df = self.df[self.df["label"] >= 1].reset_index(drop=True)
```

All Phase 2 training scripts (M2, M3, M4) pass `exclude_grade0=True` for severity_train, severity_val, and severity_test datasets.

M1 does NOT use this flag.

**Important interaction with availability matrices:** Option A (recommended) — regenerate availability matrices after filtering (see `generate_availability.py` below). This ensures matrix row indices align directly with the filtered dataset, eliminating all `val_sev_mask` / `sev_mask` runtime pre-filtering from v4.

---

### 6. `src/evaluate.py` — MINOR MODIFY

The `severity_only=True` default is already correct from v4. Update the docstring only: grade-0 exclusion is now by design, not a cohort-confound workaround.

> "Phase 2 operates on grades 1–4 only. Grade-0 patients are handled by M1 (Phase 1) and are not part of the scheduling problem."

---

### 7. `src/generate_availability.py` — MODIFY

Filter to grades 1–4 before counting N for each split.

```python
for split_name, seed in splits.items():
    split_df = manifest[manifest["split"] == split_name]
    split_df = split_df[split_df["label"] >= 1]   # ← add this filter
    N = len(split_df)
    avail = simulate_availability(N, T, p_available, seed=seed)
    ...
```

This eliminates the `val_sev_mask` / `sev_mask` pre-filtering pattern from v4 — the matrices are directly sized for the filtered datasets.

---

### 8. `config.py` — MODIFY

Remove the two-phase training parameters that no longer apply.

```python
# V5: Feature-freezing decomposition
# Phase 1 (M1): trains full network on binary labels
# Phase 2 (M2/M3/M4): only severity_head is trainable; backbone + trunk frozen from M1

# Remove:
# "epochs_phase1"    — no longer needed; Phase 2 is single-phase
# "lr_finetune"      — backbone is never unfrozen in Phase 2
# "lr_trunk_phase2"  — trunk is never unfrozen in Phase 2

# Keep:
"lr_head": 1e-4,         # learning rate for severity_head (M2, M3, Stage 2 of M4)
"lr_stage3": 1e-5,       # learning rate for DFL fine-tuning (M4 Stage 3)
"epochs_stage2": 30,     # CE training epochs for severity_head
"epochs_stage3": 20,     # DFL fine-tuning epochs
```

---

## Implementation Order

1. **`src/generate_availability.py`** — Add grade ≥ 1 filter. Regenerate all three `.npy` files.
2. **`src/dataset.py`** — Add `exclude_grade0` flag.
3. **`src/train_M2.py`** (rename from `train_M2a.py`) — Single-phase, frozen backbone+trunk, severity head only, grades 1–4. Verify it trains.
4. **`src/train_M3.py`** (rename from `train_M2b.py`) — Same as M2 + scheduling cost checkpoint selection. Verify.
5. **`src/train_M4.py`** (rename from `train_M3.py`) — Stage 2 = M3, Stage 3 = DFL on severity head only. Verify DFL loss decreases.
6. **`src/evaluate.py`** — Update comments/docstrings only.
7. **`config.py`** — Remove obsolete two-phase parameters.

---

## Sanity Checks

| Check | Expected |
|---|---|
| Trainable params in Phase 2 (any method) | Only `severity_head.*` parameters |
| Grade-0 rows in Phase 2 training data | Zero |
| Grade-0 rows in Phase 2 validation data | Zero |
| Grade-0 rows in evaluation | Zero |
| Availability matrix shapes | Match filtered (grade ≥ 1) dataset sizes |
| M2 and M3 produce different checkpoints | Yes — different selection criteria may pick different epochs |
| M4 Stage 2 checkpoint | Identical training procedure to M3 |
| M4 DFL training loss | Should decrease across epochs; if flat, check sigma and batch size |
| All models evaluated with same test availability matrix | Yes — loaded from same `.npy` file |

---

## Files Summary

| File | Action | Key Change |
|---|---|---|
| `src/generate_availability.py` | Modify | Filter to grades 1–4 before counting rows |
| `src/dataset.py` | Modify | Add `exclude_grade0` flag |
| `src/train_M1.py` | No change | Phase 1 — unchanged |
| `src/train_M2a.py` → `src/train_M2.py` | Rename + modify | Freeze backbone+trunk, single-phase, grades 1–4 only, val CE |
| `src/train_M2b.py` → `src/train_M3.py` | Rename + modify | Same as M2 + val scheduling cost checkpoint selection |
| `src/train_M3.py` → `src/train_M4.py` | Rename + modify | Stage 2 = M3, Stage 3 = DFL on severity head only |
| `src/evaluate.py` | Minor modify | Update comments only |
| `config.py` | Modify | Remove obsolete two-phase params |
| `src/model.py` | No change | — |
| `src/losses.py` | No change | — |
| `src/allocation.py` | No change | — |
| `src/simulate_availability.py` | No change | — |
| `tests/test_solver_availability.py` | No change | — |
