# V4 Implementation Spec: Availability-Constrained Scheduling

All decisions are final. Implement exactly as written below.

---

## Files to Touch

| File | Action |
|---|---|
| `src/simulate_availability.py` | **Create new** |
| `src/solver.py` | **Add** `solve_multislot_availability()` — do not touch existing code |
| `src/evaluate.py` | **Modify** — pass availability to solver |
| `src/train_M2b.py` | **Modify** — use availability-constrained val cost for checkpoint selection |
| `src/train_M3.py` | **Modify** — re-sample availability per batch in training loop |
| `configs/config.yaml` | **Modify** — add availability settings |
| `tests/test_solver_availability.py` | **Create new** |
| `src/train_M1.py` | Do not touch |
| `src/train_M2a.py` | Do not touch |
| `src/model.py` | Do not touch |
| `src/cost.py` | Do not touch |

---

## 1. New File: `src/simulate_availability.py`

```python
import numpy as np

def simulate_availability(N, T, p_available=0.7, seed=None):
    """
    Generate a binary availability matrix.

    Args:
        N:           Number of patients
        T:           Number of slots
        p_available: Bernoulli probability each patient is available per slot
        seed:        Random seed. Pass None during M3 training for stochastic sampling.

    Returns:
        availability: (N, T) binary numpy array
    """
    rng = np.random.RandomState(seed)
    availability = rng.binomial(1, p_available, size=(N, T))

    # Ensure every patient has at least one available slot
    for i in range(N):
        if availability[i].sum() == 0:
            slot = rng.randint(T)
            availability[i, slot] = 1

    return availability
```

**Also create a one-time generation script** `scripts/generate_availability.py`:

```python
"""
Run once before any training to generate and save fixed val/test availability matrices.
"""
import numpy as np
from src.simulate_availability import simulate_availability
import os

os.makedirs('data/availability', exist_ok=True)

N_val, N_test, T = ...  # fill in from your dataset
seed_val  = 100
seed_test = 200

val_availability  = simulate_availability(N_val,  T, p_available=0.7, seed=seed_val)
test_availability = simulate_availability(N_test, T, p_available=0.7, seed=seed_test)

np.save(f'data/availability/val_availability_seed{seed_val}.npy',   val_availability)
np.save(f'data/availability/test_availability_seed{seed_test}.npy', test_availability)

print("Saved val and test availability matrices.")
```

---

## 2. Modified File: `src/solver.py`

Add the function below. **Do not modify any existing functions.**

```python
def solve_multislot_availability(scores, K_list, availability):
    """
    Greedy multi-slot assignment with availability constraints.

    Args:
        scores:       (N,) array — higher score = more urgent
        K_list:       List of T ints, capacity per slot
        availability: (N, T) binary matrix; 1 means patient i can go in slot t

    Returns:
        z: (N, T) binary assignment matrix

    Slot processing order: t=0 first, t=T-1 last (increasing index).
    With delay=[1.0, 3.0, 8.0], this fills the lowest-delay slot first.
    """
    import numpy as np
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

---

## 3. New File: `tests/test_solver_availability.py`

```python
import numpy as np
from src.solver import solve_multislot, solve_multislot_availability

def test_all_ones_matches_unconstrained():
    """
    When availability = all ones, solve_multislot_availability must
    produce the exact same result as solve_multislot.
    """
    np.random.seed(0)
    N, T = 100, 3
    scores = np.random.rand(N)
    K_list = [5, 10, 15]
    availability = np.ones((N, T), dtype=int)

    z_unconstrained = solve_multislot(scores, K_list)
    z_constrained   = solve_multislot_availability(scores, K_list, availability)

    assert np.array_equal(z_unconstrained, z_constrained)

def test_unavailable_patient_not_assigned():
    """Patient with all-zero availability must not be assigned to any slot."""
    N, T = 50, 3
    scores = np.ones(N)
    K_list = [10, 10, 10]
    availability = np.ones((N, T), dtype=int)
    availability[0, :] = 0   # patient 0 fully unavailable

    z = solve_multislot_availability(scores, K_list, availability)
    assert z[0, :].sum() == 0, "Fully unavailable patient must not be assigned"

def test_total_assignments_within_capacity():
    """Total assignments must never exceed sum of K_list."""
    np.random.seed(42)
    N, T = 200, 3
    scores = np.random.rand(N)
    K_list = [10, 20, 30]
    availability = np.random.binomial(1, 0.5, size=(N, T))

    z = solve_multislot_availability(scores, K_list, availability)
    assert z.sum() <= sum(K_list)
```

---

## 4. Modified File: `src/evaluate.py`

Update the evaluate function to accept and use an availability matrix.
`scheduling_cost()` is unchanged — only the solver call changes.

```python
from src.solver import solve_multislot_availability
import numpy as np

def evaluate(model, X, Y, K_list, availability):
    """
    Args:
        model:        Trained model with a .predict(X) method
        X:            Input features
        Y:            True severity labels
        K_list:       List of slot capacities
        availability: (N, T) binary matrix — must be the same for all models
                      being compared

    Returns:
        cost: scalar scheduling cost
    """
    scores = model.predict(X)
    z = solve_multislot_availability(scores, K_list, availability)
    cost = scheduling_cost(z, Y)
    return cost
```

**Calling pattern** (in your experiment runner):

```python
test_availability = np.load('data/availability/test_availability_seed200.npy')

# Pass the same matrix to every model — required for fair comparison
for model in [m1, m2a, m2b, m3]:
    cost = evaluate(model, X_test, Y_test, K_list, test_availability)
```

### C_random and C_oracle Must Also Use Availability Constraints

`C_norm = C_total / C_random` is the primary reported metric. Both the random baseline
and the oracle **must use `solve_multislot_availability` with the same availability matrix**
as the models. If they use the unconstrained solver, the random baseline gets "free"
assignments that no model can make, making `C_norm` meaningless.

```python
def compute_baselines(Y, K_list, availability, n_random_seeds=100):
    """
    Compute C_random and C_oracle under availability constraints.
    Both use the same availability matrix as the models being evaluated.

    Args:
        Y:            (N,) true severity labels
        K_list:       List of slot capacities
        availability: (N, T) binary matrix — same one used for model evaluation
        n_random_seeds: number of random score draws to average for C_random

    Returns:
        c_random: average cost of random scoring under availability
        c_oracle: cost of perfect scoring (uses true severity as scores) under availability
    """
    N = len(Y)

    # C_oracle: use true severity as scores — best achievable by a perfect model
    oracle_scores = Y.astype(float)
    z_oracle = solve_multislot_availability(oracle_scores, K_list, availability)
    c_oracle = scheduling_cost(z_oracle, Y)

    # C_random: average over many random score draws
    random_costs = []
    for seed in range(n_random_seeds):
        rng = np.random.RandomState(seed)
        random_scores = rng.rand(N)
        z_rand = solve_multislot_availability(random_scores, K_list, availability)
        random_costs.append(scheduling_cost(z_rand, Y))
    c_random = np.mean(random_costs)

    return c_random, c_oracle

# Usage — run once per seed, reuse for all model comparisons
test_availability = np.load('data/availability/test_availability_seed200.npy')
c_random, c_oracle = compute_baselines(Y_test, K_list, test_availability)

for model in [m1, m2a, m2b, m3]:
    c_total = evaluate(model, X_test, Y_test, K_list, test_availability)
    c_norm  = c_total / c_random
```

---

## 5. Modified File: `src/train_M2b.py`

Only the checkpoint selection criterion changes. The training loop itself is unchanged.

```python
from src.simulate_availability import simulate_availability
from src.solver import solve_multislot_availability
import numpy as np

# Load fixed val availability once before training
val_availability = np.load('data/availability/val_availability_seed100.npy')

best_val_cost = float('inf')

for epoch in range(epochs_M2):
    # --- training step: unchanged ---

    # Checkpoint selection: use availability-constrained scheduling cost
    val_scores = model.predict(X_val)
    z_val      = solve_multislot_availability(val_scores, K_list, val_availability)
    val_cost   = scheduling_cost(z_val, Y_val)

    if val_cost < best_val_cost:
        best_val_cost = val_cost
        save_checkpoint(model, epoch)
```

---

## 6. Modified File: `src/train_M3.py`

Two changes: (1) re-sample availability **once per batch** during training, (2) use fixed val availability for checkpoint selection.

```python
from src.simulate_availability import simulate_availability
from src.solver import solve_multislot_availability
import numpy as np
import torch

# Load fixed val availability once before training
val_availability = np.load('data/availability/val_availability_seed100.npy')

best_val_cost = float('inf')

for epoch in range(epochs_M3):
    for batch in dataloader:
        X_batch, Y_batch, has_severity = batch

        # The solver operates only on the severity-labeled subset of the batch.
        # CRITICAL: avail_batch must be sized for n_sev, not len(X_batch).
        mask  = has_severity & (Y_batch >= 0)
        X_sev = X_batch[mask]
        Y_sev = Y_batch[mask]
        n_sev = len(X_sev)

        scores = model(X_sev)             # (n_sev,)

        # Sample a fresh availability matrix for this batch.
        # seed=None means it is random each time (intentional).
        # Size is n_sev — the severity-labeled subset, NOT len(X_batch).
        avail_batch = simulate_availability(
            N=n_sev, T=T, p_available=0.7, seed=None
        )

        # Randomized smoothing gradient estimation.
        # avail_batch is SHARED across all M perturbations —
        # we want to measure "what changes when scores change,
        # holding constraints fixed."
        costs = []
        xis   = []
        for m in range(M_perturbations):
            xi = torch.randn_like(scores) * sigma
            perturbed_scores = (scores + xi).detach().numpy()

            z_m    = solve_multislot_availability(perturbed_scores, K_list, avail_batch)
            cost_m = scheduling_cost(z_m, Y_sev.numpy())

            costs.append(cost_m)
            xis.append(xi)

        baseline = np.mean(costs)

        # REINFORCE-style gradient: (1 / M*sigma) * sum((cost_m - baseline) * xi_m)
        grad = sum(
            (cost_m - baseline) * xi_m
            for cost_m, xi_m in zip(costs, xis)
        ) / (M_perturbations * sigma)

        scores.backward(grad / n_sev)
        optimizer.step()
        optimizer.zero_grad()

    # Checkpoint selection: same logic as M2b
    val_scores = model.predict(X_val)
    z_val      = solve_multislot_availability(val_scores, K_list, val_availability)
    val_cost   = scheduling_cost(z_val, Y_val)

    if val_cost < best_val_cost:
        best_val_cost = val_cost
        save_checkpoint(model, epoch)
```

---

## 7. Modified File: `configs/config.yaml`

Add these lines under the scheduling problem section:

```yaml
# Availability constraints
p_available: 0.7
availability_seeds:
  val: 100
  test: 200
```

---

## 8. Implementation Order

Do these in order — each step validates the previous one:

1. Write `simulate_availability.py` and verify it runs.
2. Add `solve_multislot_availability()` to `solver.py`. Run `test_all_ones_matches_unconstrained()` — it must pass before proceeding.
3. Run the generation script to save `val_availability_seed100.npy` and `test_availability_seed200.npy`.
4. Update `evaluate.py`. Smoke-test by evaluating any already-trained model checkpoint.
5. Update `train_M2b.py`. Retrain M2b.
6. Update `train_M3.py`. Retrain M3.

---

## 9. Sanity Checks

| Check | Expected |
|---|---|
| `solve_multislot_availability(scores, K_list, ones_matrix)` | Identical output to `solve_multislot(scores, K_list)` |
| Patient with all-zero availability row | Never appears in `z` |
| `z.sum()` | ≤ `sum(K_list)`; may be strictly less when availability is sparse |
| M3 training loss across epochs | Should decrease; if flat, check σ, batch size, or gradient estimator |
| Val/test availability | Always loaded from saved `.npy` files — never regenerated mid-experiment |
