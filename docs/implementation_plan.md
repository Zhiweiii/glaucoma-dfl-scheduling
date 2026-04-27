# Implementation Plan: glaucoma-dfl-scheduling

Working reference for porting old repo (multislot-scheduling branch) → new repo + v4 availability constraints.

---

## Source

**Old repo:** `Glaucoma-Triage-DFL` on branch `multislot-scheduling`  
**New repo:** `glaucoma-dfl-scheduling` (currently empty except docs)

---

## Final Directory Structure

```
glaucoma-dfl-scheduling/
├── config.py                          # MODIFIED from old repo
├── pyproject.toml                     # COPY (update project name)
├── uv.lock                            # COPY
├── src/
│   ├── __init__.py                    # COPY unchanged
│   ├── model.py                       # COPY unchanged
│   ├── dataset.py                     # COPY unchanged
│   ├── losses.py                      # COPY unchanged
│   ├── allocation.py                  # MODIFIED: add solve_multislot_availability()
│   ├── simulate_availability.py       # NEW
│   ├── evaluate.py                    # MODIFIED: add availability parameter
│   ├── train_M1.py                    # RENAMED from train_M1_v3.py (minor path fix)
│   ├── train_M2a.py                   # RENAMED from train_M2a_v3.py (minor path fix)
│   ├── train_M2b.py                   # RENAMED + MODIFIED: availability val criterion
│   ├── train_M3.py                    # RENAMED + MODIFIED: availability in DFL loop
│   └── catalog/
│       ├── __init__.py                # NEW (empty)
│       ├── data_pipeline_v2.py        # COPY from src/
│       └── construct_datasets_v2.py   # COPY from src/
├── scripts/
│   └── generate_availability.py       # NEW: run once before training
├── data/
│   └── availability/                  # generated .npy files land here (gitignored)
├── tests/
│   ├── __init__.py                    # NEW (empty)
│   └── test_solver_availability.py    # NEW
└── docs/
    └── ...
```

---

## Spec Decisions (from v4_implementation_spec.md)

| Question | Answer |
|---|---|
| Slot processing order | t=0 first (lowest delay). With delay=[1,3,8], fills least-urgent slot first. Matches v3 behavior, so `all_ones` regression test passes. |
| M3 availability sampling | Per-batch. `seed=None` → stochastic each batch. Same matrix shared across all M perturbations within that batch. |
| M2b/M3 val availability | Fixed matrix loaded once from `data/availability/val_availability_seed100.npy`. Not resampled. |

---

## Naming Reconciliation (spec vs old repo)

The spec uses different names than the old repo. We follow the old repo's conventions:

| Spec name | Actual file | Notes |
|---|---|---|
| `src/solver.py` | `src/allocation.py` | add new function here |
| `src/cost.py` | `src/losses.py` | unchanged |
| `configs/config.yaml` | `config.py` | keep Python format |
| `solve_multislot` (in test) | `assign_slots` | old name, kept |

Tests should import `from src.allocation import assign_slots, solve_multislot_availability`.

---

## File-by-File Change Log

### `config.py` — MODIFY

Add three keys to CONFIG dict:

```python
"p_available":          0.7,    # Bernoulli prob each patient available per slot
"availability_seed_val":  100,  # seed for fixed val availability matrix
"availability_seed_test": 200,  # seed for fixed test availability matrix
```

### `src/allocation.py` — MODIFY

Add one new function. **Do not touch existing `assign_slots` or `make_K_list`.**

```python
def solve_multislot_availability(
    scores: np.ndarray,
    K_list: list[int],
    availability: np.ndarray,
) -> np.ndarray:
    """
    Greedy multi-slot assignment with availability constraint.
    Slots processed in increasing order (t=0 first).
    
    Args:
        scores:       (N,) float array — higher = more urgent
        K_list:       list of T ints, capacity per slot
        availability: (N, T) int array, 1 means patient i can go in slot t
    Returns:
        z: (N, T) int array
    """
    N = len(scores)
    T = len(K_list)
    z = np.zeros((N, T), dtype=int)
    assigned = np.zeros(N, dtype=bool)

    for t in range(T):
        eligible = (~assigned) & (availability[:, t] == 1)
        eligible_idx = np.where(eligible)[0]
        if len(eligible_idx) == 0:
            continue
        sorted_eligible = eligible_idx[np.argsort(-scores[eligible_idx])]
        n_assign = min(K_list[t], len(sorted_eligible))
        for idx in sorted_eligible[:n_assign]:
            z[idx, t] = 1
            assigned[idx] = True

    return z
```

**Type note:** `assign_slots` (existing) takes a torch Tensor and returns a torch Tensor — used internally in v3 training. `solve_multislot_availability` (new) is numpy in / numpy out — used in evaluate.py, M2b/M3 val criterion, and M3 DFL gradient loop. This is intentional.

**Import needed at top of file:** `import numpy as np`

### `src/simulate_availability.py` — NEW

```python
import numpy as np

def simulate_availability(N, T, p_available=0.7, seed=None):
    rng = np.random.RandomState(seed)
    availability = rng.binomial(1, p_available, size=(N, T))
    for i in range(N):
        if availability[i].sum() == 0:
            slot = rng.randint(T)
            availability[i, slot] = 1
    return availability
```

### `src/evaluate.py` — MODIFY

Current `evaluate()` signature:
```python
def evaluate(predictions_csv, alpha, delay, beta, K_frac_list, d_miss, n_random) -> dict
```

New signature — add optional `availability` parameter:
```python
def evaluate(predictions_csv, alpha, delay, beta, K_frac_list, d_miss, n_random,
             availability=None) -> dict
```

**Inside evaluate():**
- Add import: `from src.allocation import solve_multislot_availability`
- Replace `z_model = assign_slots(scores, K_frac_list)` with:
  ```python
  if availability is not None:
      z_model = solve_multislot_availability(scores, K_list, availability)
  else:
      z_model = assign_slots(scores, K_frac_list)
  ```
- Do the same for `oracle_cost` and `random_cost` — both need availability for
  fair comparison (C_norm = C_total / C_random is meaningless if random is unconstrained
  but model is constrained).

**oracle_cost**: add `availability=None`, pass to solver.  
**random_cost**: add `availability=None`, pass to solver. Random assignment must also
respect availability (draw uniformly from feasible assignments, not all permutations).  

**CLI**: add `--availability PATH` arg (path to .npy file; defaults to None = unconstrained).

**Backward compatibility**: `availability=None` preserves all existing behavior. 
Training scripts that auto-evaluate (train_M1, train_M2a) call `evaluate()` without
availability — they get unconstrained evaluation, which is fine (they don't use availability
in training either; availability only affects M2b val criterion and M3 training + all test eval).

### `src/train_M1.py` — RENAME only

Copy from `train_M1_v3.py`. Changes:
- Script name references: update docstring/print to say `train_M1.py` (not `train_M1_v3.py`)
- Checkpoint/result file names: `M1_v3_seed{seed}.pt` → `M1_seed{seed}.pt`, etc.
- Auto-evaluate call: no availability arg (M1 doesn't use it during training)

### `src/train_M2a.py` — RENAME only

Same as M1: rename, update output file names (`M2a_v3_*` → `M2a_*`), no availability changes.

### `src/train_M2b.py` — MODIFY

In addition to rename / output file name updates:

1. Add imports at top:
   ```python
   import numpy as np
   from src.simulate_availability import simulate_availability
   from src.allocation import solve_multislot_availability
   ```

2. `val_decision_cost` signature change:
   ```python
   def val_decision_cost(model, loader, alpha, beta, K_frac_list, delay, d_miss, device,
                         val_availability: np.ndarray) -> float:
   ```
   Inside the function, replace:
   ```python
   z = assign_slots(scores, K_list)
   return scheduling_cost_multislot(z, labels, alpha.cpu(), beta, delay.cpu(), d_miss).item()
   ```
   with:
   ```python
   scores_np = scores.numpy()
   z_np = solve_multislot_availability(scores_np, K_list, val_availability)
   z = torch.tensor(z_np, dtype=alpha.dtype)
   return scheduling_cost_multislot(z, labels, alpha.cpu(), beta, delay.cpu(), d_miss).item()
   ```

3. In `train_M2b()`, load val_availability once before the training loop:
   ```python
   val_avail_path = Path("data/availability") / f"val_availability_seed{CONFIG['availability_seed_val']}.npy"
   val_availability = np.load(val_avail_path)
   ```
   Then pass `val_availability=val_availability` to every call to `val_decision_cost`.

4. Output file names: `M2b_v3_*` → `M2b_*`

### `src/train_M3.py` — MODIFY

All M2b changes above, plus two additions to `dfl_step`:

1. New parameter: `p_available: float` (or read from CONFIG inside the function)

2. Inside `dfl_step`, before the perturbation loop, add:
   ```python
   T = len(K_list)
   avail_batch = simulate_availability(N, T, p_available=CONFIG["p_available"], seed=None)
   ```

3. Inside the perturbation loop, replace:
   ```python
   z_m = assign_slots(perturbed, K_list)
   ```
   with:
   ```python
   perturbed_np = perturbed.cpu().numpy()
   z_m_np = solve_multislot_availability(perturbed_np, K_list, avail_batch)
   z_m = torch.tensor(z_m_np, dtype=alpha.dtype, device=device)
   ```

4. Output file names: `M3_v3_*` → `M3_*`

---

## New Files

### `scripts/generate_availability.py`

Run once before any training to generate fixed val/test availability matrices.
Must load the manifest to get N_val and N_test:

```python
"""Run once before training: saves val and test availability matrices."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.simulate_availability import simulate_availability

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--p-available", type=float, default=0.7)
    p.add_argument("--out-dir", default="data/availability")
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    T = 3  # number of slots

    splits = {
        "val":  ("severity_val",  100),
        "test": ("severity_test", 200),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (split_name, seed) in splits.items():
        rows = df[df["split"] == split_name]
        N = len(rows)
        avail = simulate_availability(N, T, args.p_available, seed=seed)
        path = out_dir / f"{name}_availability_seed{seed}.npy"
        np.save(path, avail)
        print(f"Saved {name} availability: shape={avail.shape} → {path}")

if __name__ == "__main__":
    main()
```

### `tests/test_solver_availability.py`

```python
import numpy as np
import pytest
from src.allocation import assign_slots, solve_multislot_availability
import torch

def test_all_ones_matches_unconstrained():
    np.random.seed(0)
    N, T = 100, 3
    scores_np = np.random.rand(N)
    K_list = [5, 10, 15]
    availability = np.ones((N, T), dtype=int)

    z_constrained = solve_multislot_availability(scores_np, K_list, availability)

    # compare against assign_slots (torch-based)
    scores_t = torch.tensor(scores_np, dtype=torch.float32)
    z_unconstrained = assign_slots(scores_t, K_list).numpy().astype(int)

    assert np.array_equal(z_unconstrained, z_constrained)

def test_unavailable_patient_not_assigned():
    N, T = 50, 3
    scores = np.ones(N)
    K_list = [10, 10, 10]
    availability = np.ones((N, T), dtype=int)
    availability[0, :] = 0

    z = solve_multislot_availability(scores, K_list, availability)
    assert z[0].sum() == 0

def test_total_assignments_within_capacity():
    np.random.seed(42)
    N, T = 200, 3
    scores = np.random.rand(N)
    K_list = [10, 20, 30]
    availability = np.random.binomial(1, 0.5, size=(N, T))

    z = solve_multislot_availability(scores, K_list, availability)
    assert z.sum() <= sum(K_list)

def test_each_patient_assigned_at_most_once():
    np.random.seed(7)
    N, T = 100, 3
    scores = np.random.rand(N)
    K_list = [10, 10, 10]
    availability = np.random.binomial(1, 0.7, size=(N, T))

    z = solve_multislot_availability(scores, K_list, availability)
    assert (z.sum(axis=1) <= 1).all()
```

---

## Random/Oracle Baseline with Availability — CONFIRMED by spec

C_random and C_oracle **must** use `solve_multislot_availability` with the same availability
matrix. Spec provides a `compute_baselines(Y, K_list, availability, n_random_seeds=100)`
helper that replaces the old `random_cost` / `oracle_cost` functions.

```python
def compute_baselines(Y, K_list, availability, n_random_seeds=100):
    oracle_scores = Y.astype(float)
    z_oracle = solve_multislot_availability(oracle_scores, K_list, availability)
    c_oracle = scheduling_cost(z_oracle, Y, ...)

    random_costs = []
    for seed in range(n_random_seeds):
        rng = np.random.RandomState(seed)
        z_rand = solve_multislot_availability(rng.rand(N), K_list, availability)
        random_costs.append(scheduling_cost(z_rand, Y, ...))
    c_random = np.mean(random_costs)
    return c_random, c_oracle
```

Replaces `oracle_cost()` and `random_cost()` in `evaluate.py`. Both take `availability`
as a required arg when availability is not None; old unconstrained path (availability=None)
calls the old functions unchanged for backward compat.

---

## Implementation Order

Follow spec Section 8 strictly:

1. `src/simulate_availability.py` — pure function, no deps
2. `src/allocation.py` — add `solve_multislot_availability`; immediately run test_all_ones_matches_unconstrained
3. `scripts/generate_availability.py` — run to create .npy files
4. `src/evaluate.py` — add availability param; smoke-test with old checkpoint + unconstrained (availability=None)
5. Copy / rename M1, M2a (no logic change, just output names)
6. `src/train_M2b.py` — update val_decision_cost + load val_availability
7. `src/train_M3.py` — update dfl_step + val_decision_cost
8. Run tests/test_solver_availability.py
9. Copy catalog files to src/catalog/

---

## Edge Cases & Gotchas

- `assign_slots` (torch) vs `solve_multislot_availability` (numpy): never mix inputs.
  When calling the numpy solver from training code, always `.detach().cpu().numpy()` first,
  then convert output back with `torch.tensor(z_np, dtype=..., device=device)`.

- `scheduling_cost_multislot` (losses.py) takes torch Tensor for z. After calling the
  numpy solver, always convert: `z = torch.tensor(z_np, dtype=alpha.dtype, device=device)`.

- `val_availability` sized for severity_val rows only (NOT all val rows). The loader
  iterates over severity_val split. Confirm N matches `len(df[df["split"]=="severity_val"])`.

- `test_availability` sized for severity_test rows. Same concern.

- `avail_batch` in M3 DFL step is sized for the severity-labeled subset of the batch
  (i.e., `n_sev = mask.sum().item()`, NOT the full batch size B). Generate:
  `avail_batch = simulate_availability(n_sev, T, ...)`.

- The `data/availability/` directory must exist before `generate_availability.py` runs.
  The script creates it; or add a `.gitkeep`.

- Checkpoint file names drop `_v3`: `M1_seed{seed}.pt`, `M2a_seed{seed}.pt`, etc.
