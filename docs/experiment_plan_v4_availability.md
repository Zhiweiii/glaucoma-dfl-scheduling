# Experiment Plan v4: Availability-Constrained Multi-Slot Glaucoma Scheduling

This document extends the existing multi-slot scheduling experiment (v3) by introducing **patient-slot availability constraints**. The availability constraint transforms the problem from a pure ranking/top-K selection into a true constrained assignment problem, making the decision layer non-separable and tightly coupled with predictions.

**What changed from v3:**
- New binary availability matrix `availability[i,t]` constraining which patients can be assigned to which slots
- Greedy solver updated to filter by availability before ranking
- M3 (DFL) training updated to sample availability inside each batch
- Evaluation pipeline updated to use shared availability across all models
- Problem is now a **constrained assignment** rather than a ranking

---

## 1. Project Goal

### 1.1 High-Level Objective

This project studies how the gap between decision-blind and decision-focused learning methods changes when the downstream optimization problem becomes more constrained and combinatorially complex. Specifically, we upgrade the glaucoma referral scheduling problem from a simple multi-slot ranking (where the solver just sorts patients by score and fills slots greedily) to an **availability-constrained assignment** problem, where each patient is only available for a subset of appointment slots.

### 1.2 Why This Matters

In the original multi-slot formulation (v3), the greedy solver processes slots sequentially and assigns the highest-scoring unassigned patients. Although the problem has multiple slots with different delay costs, the solver is effectively a sequential top-K selection — the optimal assignment is still determined purely by the ranking of scores. This means the problem is **separable**: the model only needs to produce a correct ranking, and small errors in predicted scores don't change the assignment as long as the ranking is preserved.

Adding availability constraints breaks this separability. When patient *i* is unavailable for slot *t*, the solver must skip them and consider the next-best available patient. This means:

- The **identity** of who gets assigned to each slot depends on the availability pattern, not just the score ranking.
- Two patients with similar scores may end up in very different slots (or one may be missed entirely) depending on their availability profiles.
- The model must now produce scores that lead to good assignments **under the specific availability realization**, not just a good ranking in general.
- The problem becomes **non-separable** and **decision-coupled**: the assignment of patient *i* affects what's available for patient *j*.

This is the core insight from the course (Chapters 10–11): when the downstream optimization is combinatorial and constrained, decision-focused learning should have a larger advantage over two-stage methods because the mapping from predictions to decisions is more complex and discontinuous.

### 1.3 Specific Research Questions

1. **Does the DFL advantage grow?** Compare the cost gap between M3 (DFL) and M2 (light-touch two-stage) under availability constraints versus the unconstrained baseline. The hypothesis is that the gap should be larger under constraints because the solver is more sensitive to prediction quality.

2. **How does availability sparsity affect the gap?** Sweep `p_available` from 0.5 (very constrained) to 1.0 (unconstrained, recovering v3). At `p_available = 1.0`, we should recover the v3 results. As `p_available` decreases, the problem becomes harder and DFL should increasingly outperform.

3. **Does M3 learn different representations?** Analyze whether M3 under availability constraints learns to differentiate patients who are "substitutable" (available in many slots, so ranking errors are forgivable) versus patients who are "bottleneck" (available in only one slot, so their score must be accurate).

### 1.4 Connection to Course Material

This upgrade directly instantiates the framework from the randomized smoothing and PG losses chapters (Chapters 10–11). The availability-constrained greedy solver is a combinatorial optimization layer with a piecewise-constant solution map. The M3 training uses randomized smoothing (perturb scores, solve, compute cost, estimate gradient) to differentiate through this solver. Adding availability constraints makes the solution map more discontinuous — a small change in scores can cause a cascade of reassignments — which is exactly the setting where randomized smoothing and DFL are most needed.

---

## 2. Mathematical Formulation

### 2.1 Notation (Additions to v3)

| Symbol | Meaning |
|---|---|
| `availability[i,t]` ∈ {0,1} | Whether patient *i* is available for slot *t* |
| `p_available` | Bernoulli probability for simulating availability (default 0.7) |

All other notation is unchanged from v3: *N* patients, *T* = 3 slots with capacities *K_t* = *K_frac_t* × *N*, delay weights δ = [1.0, 3.0, 8.0], miss penalty *d_miss* = 15.0, severity costs α = (0, 1, 3, 6, 10), per-referral cost β.

### 2.2 Cost Function (Unchanged from v3)

The per-patient cost function remains:

```
C(z, Y) = Σ_{i,t} z[i,t] * (α[y_i] * delay[t] + β) + Σ_i (1 - Σ_t z[i,t]) * α[y_i] * d_miss
```

where:
- `z[i,t]` ∈ {0,1} indicates patient *i* is assigned to slot *t*
- Each patient is assigned to at most one slot: `Σ_t z[i,t]` ≤ 1
- Slot capacity: `Σ_i z[i,t]` ≤ `K_t`

### 2.3 New Constraint

```
z[i,t] ≤ availability[i,t]    ∀ i, t
```

Patient *i* can only be assigned to slot *t* if `availability[i,t] = 1`. This is the only change to the optimization problem.

### 2.4 Full Optimization Problem

```
min_{z}   C(z, Y)
s.t.      Σ_t z[i,t] ≤ 1           ∀ i        (each patient assigned at most once)
          Σ_i z[i,t] ≤ K_t         ∀ t        (slot capacity)
          z[i,t] ≤ availability[i,t] ∀ i,t     (availability constraint)  [NEW]
          z[i,t] ∈ {0,1}           ∀ i,t
```

---

## 3. Implementation Plan

### 3.1 Phase 0: Availability Simulation

**File:** `simulate_availability.py`

**Purpose:** Generate and manage availability matrices for train/val/test splits.

**Implementation details:**

```python
def simulate_availability(N, T, p_available=0.7, seed=None):
    """
    Generate availability matrix.
    
    Args:
        N: number of patients
        T: number of slots
        p_available: probability each patient is available for each slot
        seed: random seed for reproducibility
    
    Returns:
        availability: (N, T) binary matrix
    """
    rng = np.random.RandomState(seed)
    availability = rng.binomial(1, p_available, size=(N, T))
    
    # CRITICAL: ensure each patient has at least one available slot
    # If a patient has all zeros, randomly set one slot to 1
    for i in range(N):
        if availability[i].sum() == 0:
            slot = rng.randint(T)
            availability[i, slot] = 1
    
    return availability
```

**Key design decisions:**
- The availability matrix is generated **once per split** (train/val/test) and **shared across all models**. This ensures fair comparison.
- For the **training set** used by M3, availability is re-sampled each epoch (or each batch, see Section 3.4) to provide diverse constraint patterns during training.
- For **val and test**, availability is fixed and saved to disk. All models are evaluated on the same availability realization.
- Seed management: use `seed = base_seed + split_id` where `split_id` ∈ {0=train, 1=val, 2=test}.

**Storage format:** Save as `.npy` files:
```
data/availability/
  val_availability_seed{s}.npy      # (N_val, T)
  test_availability_seed{s}.npy     # (N_test, T)
```

### 3.2 Phase 1: Solver Update

**File:** Modify existing `solver.py` → add `solve_multislot_availability()`

**Current solver (v3):**
```python
def solve_multislot(scores, K_list):
    """
    Greedy multi-slot assignment without availability.
    For each slot t (in order):
        1. Filter to unassigned patients
        2. Sort by score descending
        3. Assign top K_t
    """
```

**New solver (v4):**
```python
def solve_multislot_availability(scores, K_list, availability):
    """
    Greedy multi-slot assignment WITH availability constraint.
    
    Args:
        scores: (N,) array of triage scores (higher = more urgent)
        K_list: list of T integers, capacity per slot
        availability: (N, T) binary matrix
    
    Returns:
        z: (N, T) binary assignment matrix
    
    For each slot t (in order of DECREASING delay, i.e. most urgent first):
        1. Filter to unassigned patients
        2. Keep only those with availability[i,t] = 1
        3. Sort by score descending
        4. Assign top K_t (or fewer if not enough available patients)
    """
    N = len(scores)
    T = len(K_list)
    z = np.zeros((N, T), dtype=int)
    assigned = np.zeros(N, dtype=bool)
    
    for t in range(T):
        # Eligible: unassigned AND available for this slot
        eligible = (~assigned) & (availability[:, t] == 1)
        eligible_idx = np.where(eligible)[0]
        
        if len(eligible_idx) == 0:
            continue
        
        # Sort eligible patients by score, descending
        sorted_eligible = eligible_idx[np.argsort(-scores[eligible_idx])]
        
        # Assign top K_t
        n_assign = min(K_list[t], len(sorted_eligible))
        for idx in sorted_eligible[:n_assign]:
            z[idx, t] = 1
            assigned[idx] = True
    
    return z
```

**Slot ordering convention:** Slots are processed in the order given by `K_list` and `delay`. The guide says "for each slot t" without specifying order. We process slot 0 first (delay=1.0, least urgent), then slot 1 (delay=3.0), then slot 2 (delay=8.0, most urgent). This matches the v3 convention. Alternatively, processing most-urgent first might yield different greedy solutions — **clarification needed from the user on this**.

**Unit test:** For `availability = all_ones`, the new solver must produce identical output to the old solver. Write a test confirming this.

### 3.3 Phase 2: Evaluation Pipeline Update

**File:** Modify existing `evaluate.py`

**Current evaluation (v3):**
```python
scores = model.predict(X_test)
z = solve_multislot(scores, K_list)
cost = scheduling_cost(z, Y_test)
```

**New evaluation (v4):**
```python
scores = model.predict(X_test)
availability = np.load(f'data/availability/test_availability_seed{seed}.npy')
z = solve_multislot_availability(scores, K_list, availability)
cost = scheduling_cost(z, Y_test)
```

**Critical rule:** The same availability matrix is used for **all four models** (M1, M2a, M2b, M3) within each seed. This isolates the effect of the learning method from the availability realization.

**The `scheduling_cost()` function is unchanged.** It takes the assignment matrix `z` and true labels `Y` and computes the cost as before. The availability constraint affects only which assignments are feasible, not how costs are computed.

### 3.4 Phase 3: M1, M2a, M2b — No Training Changes

The training procedures for M1 (binary-only), M2a (severity from ImageNet), and M2b (warm-start + light-touch) are **completely unchanged**. These models are decision-blind or light-touch — they don't interact with the solver during training.

The only change is in **evaluation**: their predictions are now fed through the availability-constrained solver instead of the unconstrained one.

For M2b (light-touch), the validation criterion uses the availability-constrained solver:
```python
# M2b checkpoint selection (light-touch):
# For each epoch checkpoint, compute validation decision cost
# using the availability-constrained solver
val_scores = model.predict(X_val)
val_availability = np.load(f'data/availability/val_availability_seed{seed}.npy')
z_val = solve_multislot_availability(val_scores, K_list, val_availability)
val_cost = scheduling_cost(z_val, Y_val)
# Select checkpoint with lowest val_cost
```

### 3.5 Phase 4: M3 (DFL) Training Update

This is the most substantive change. M3 uses randomized smoothing to differentiate through the solver. With availability constraints, the solver call inside the perturbation loop must include the availability matrix.

**Current M3 training loop (v3, per batch):**
```python
for batch in dataloader:
    X_batch, Y_batch = batch
    scores = model(X_batch)                          # (B,)
    
    # Randomized smoothing gradient estimation
    total_grad = 0
    costs = []
    for m in range(M_perturbations):
        xi = torch.randn_like(scores) * sigma
        perturbed_scores = scores + xi
        z_m = solve_multislot(perturbed_scores.detach().numpy(), K_list)
        cost_m = scheduling_cost(z_m, Y_batch.numpy())
        costs.append(cost_m)
    
    baseline = np.mean(costs)
    # ... compute gradient using REINFORCE-style estimator with baseline
```

**New M3 training loop (v4, per batch):**
```python
for batch in dataloader:
    X_batch, Y_batch = batch
    scores = model(X_batch)                          # (B,)
    
    # Sample availability for this batch
    # KEY DECISION: re-sample availability each batch during training
    avail_batch = simulate_availability(len(X_batch), T, p_available=0.7, seed=None)
    
    # Randomized smoothing gradient estimation
    total_grad = 0
    costs = []
    for m in range(M_perturbations):
        xi = torch.randn_like(scores) * sigma
        perturbed_scores = scores + xi
        z_m = solve_multislot_availability(
            perturbed_scores.detach().numpy(), K_list, avail_batch
        )
        cost_m = scheduling_cost(z_m, Y_batch.numpy())
        costs.append(cost_m)
    
    baseline = np.mean(costs)
    # ... compute gradient using REINFORCE-style estimator with baseline
    # Same gradient estimator as v3: 
    # ∇ ≈ (1/Mσ) Σ_m (cost_m - baseline) * xi_m
```

**Key design decisions for M3 training:**

1. **Availability is re-sampled each batch (not fixed).** During training, we want M3 to learn a model that performs well across diverse availability patterns, not overfit to one realization. This is analogous to data augmentation — each batch sees a different constraint landscape.

2. **Same availability for all M perturbations within a batch.** Within a single gradient estimation step, all M perturbed solutions use the same availability matrix. This ensures the gradient estimator is comparing "what happens when I change the scores" under the same constraints, which is the correct counterfactual.

3. **Batch size considerations.** In v3, the batch could be the full training set (since top-K is global). With availability, this is still the case — the solver operates on all patients in the batch simultaneously. If batch size < N_train, then K_list should be scaled proportionally: `K_batch_t = round(K_frac_t * B)`.

### 3.6 Phase 5: Experiments

#### Experiment 1: Main Comparison (Same as v3, but with Availability)

Run all four models (M1, M2a, M2b, M3) × 5 seeds with `p_available = 0.7`.

**Report:** Normalized cost `C_norm = C_total / C_random` for each model, mean ± std across seeds.

**Compare to v3 results:** The key comparison is the **gap** M2b − M3 (light-touch vs. DFL) under availability constraints vs. without. If the gap is larger under constraints, this supports the hypothesis that DFL benefits more when the optimization is harder.

#### Experiment 2: Availability Sparsity Sweep

Fix the best seed and sweep `p_available` ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.

At `p_available = 1.0`, recover the unconstrained v3 results (all patients available everywhere).

**Plot:** X-axis = `p_available`, Y-axis = normalized cost. One line per model. Expect the lines to diverge as `p_available` decreases (problem gets harder).

**Note:** For each value of `p_available`, generate a fresh availability matrix for val and test (but keep it fixed across models). Use the same random seed for availability generation across the sweep so the matrices are nested: the `p_available = 0.6` matrix is a subset of the `p_available = 0.8` matrix (roughly).

#### Experiment 3: Severity Label Scarcity (Same as v3 Exp 2)

Keep `p_available = 0.7` fixed. Sweep severity label fraction ∈ {0.1, 0.25, 0.5, 0.75, 1.0}.

This tests the "Cheap Thrills" hypothesis under constraints. If DFL can leverage the cost structure to compensate for scarce severity labels, this advantage should be amplified when the optimization is harder (availability-constrained).

---

## 4. File Structure

```
project/
├── data/
│   ├── availability/
│   │   ├── val_availability_p0.7_seed42.npy
│   │   ├── test_availability_p0.7_seed42.npy
│   │   └── ...
│   ├── binary/                    # unchanged
│   └── severity/                  # unchanged
├── src/
│   ├── simulate_availability.py   # NEW
│   ├── solver.py                  # MODIFIED: add solve_multislot_availability()
│   ├── cost.py                    # UNCHANGED: scheduling_cost()
│   ├── train_M1.py                # UNCHANGED
│   ├── train_M2a.py               # UNCHANGED (eval uses new solver)
│   ├── train_M2b.py               # MODIFIED: val criterion uses new solver
│   ├── train_M3.py                # MODIFIED: availability in training loop
│   ├── evaluate.py                # MODIFIED: uses new solver
│   └── model.py                   # UNCHANGED: DualHeadVGG19
├── configs/
│   └── config.yaml                # Add p_available parameter
├── results/
│   ├── v3_unconstrained/          # existing results
│   └── v4_availability/           # new results
└── tests/
    └── test_solver_availability.py # NEW: unit tests
```

---

## 5. Configuration

### 5.1 Shared Settings (Same as v3 + New)

```yaml
# Scheduling problem
T: 3
K_frac_list: [0.05, 0.10, 0.15]      # capacity fractions per slot
delay: [1.0, 3.0, 8.0]                # delay cost per slot
d_miss: 15.0                          # miss penalty
beta: 1.0                             # per-referral cost
alpha: [0, 1, 3, 6, 10]              # severity cost by grade

# NEW: Availability
p_available: 0.7                      # default availability probability
availability_seeds:                   # separate from model seeds
  val: 100
  test: 200

# Model training (unchanged)
backbone: VGG19
pretrained: ImageNet
optimizer: Adam
lr: 1e-4
batch_size: 32
epochs_M1: 30
epochs_M2: 30
epochs_M3: 20

# M3 DFL settings (unchanged)
M_perturbations: 20
sigma: 0.5
use_baseline: true
```

---

## 6. Implementation Order

This is the recommended order of implementation to minimize debugging:

1. **`simulate_availability.py`** — Pure function, no dependencies. Write + test immediately.

2. **`solve_multislot_availability()`** — Add to `solver.py`. Write unit test confirming that `availability = all_ones` recovers the old solver output.

3. **`evaluate.py`** — Update to accept an `availability` argument. Re-run evaluation on existing v3 model checkpoints with `availability = all_ones` to confirm cost matches v3 results (regression test).

4. **`evaluate.py` with real availability** — Generate test availability at `p_available = 0.7`, re-evaluate existing v3 checkpoints. Costs should increase (harder problem). This gives a "free" baseline without retraining anything.

5. **`train_M2b.py`** — Update validation criterion to use availability-constrained solver. Retrain M2b.

6. **`train_M3.py`** — Update training loop to sample availability per batch. Retrain M3.

7. **Run Experiment 1** — Full comparison at `p_available = 0.7`.

8. **Run Experiment 2** — Availability sparsity sweep.

9. **Run Experiment 3** — Severity scarcity sweep (if time permits).

---

## 7. Sanity Checks and Debugging Guide

### 7.1 Solver Correctness
- `solve_multislot_availability(scores, K_list, ones_matrix)` must equal `solve_multislot(scores, K_list)` for any scores.
- If `availability[i, :] = [0, 0, ..., 1]` (patient only available in last slot), they should only appear in the last slot or be unassigned.
- Total assignments ≤ `Σ_t K_t`. With tight availability, total assignments may be strictly less than `Σ_t K_t` (some slots may not fill).

### 7.2 Cost Sanity
- At `p_available = 1.0`, costs should match v3 exactly.
- At `p_available = 0.7`, costs should be weakly higher than v3 (fewer feasible assignments → more missed patients → higher cost).
- At `p_available = 0.5`, costs should be notably higher.
- Cost should be monotonically non-increasing in `p_available` (more availability → more flexibility → lower or equal cost).

### 7.3 M3 Training Sanity
- M3 training loss should decrease across epochs (the model is learning to produce scores that lead to lower cost under availability constraints).
- If M3 training loss is flat, check: (a) sigma too large or too small, (b) batch size too small for the solver to differentiate, (c) gradient estimator bug.

### 7.4 Reproducibility
- Availability matrices for val/test must be saved and loaded, not regenerated on the fly. If regenerated with a different seed, results will be inconsistent across runs.

---

## 8. Open Questions for Clarification

Before implementing, the following decisions should be confirmed:

1. **Slot processing order in the greedy solver:** The guide says "for each slot t" but doesn't specify order. Currently we process slot 0 (least urgent, delay=1.0) first. Should we instead process slot 2 (most urgent, delay=8.0) first? Processing the most urgent slot first would prioritize placing the highest-scoring patients in the most urgent slot, which might be more natural. The choice affects the greedy solution and potentially the DFL gap.

2. **Availability re-sampling frequency in M3 training:** The guide says "sample availability matrix" inside each batch. Should this be per-batch (our current plan) or per-epoch? Per-batch provides more diversity but means the model never sees the same constraint pattern twice. Per-epoch means the model trains on one availability pattern for all batches in that epoch, then gets a new one.

3. **Should M2b's validation also use re-sampled availability?** Currently we plan to fix one availability matrix for validation (same as test). But should M2b's light-touch selection instead average cost over multiple availability realizations? This would be more robust but more expensive.

4. **Batch size for M3 with availability:** With availability constraints, very small batches (e.g., B=32) might not have enough available patients to fill any slot, making the solver output degenerate. Should we increase batch size for M3, or use the full training set as one batch (as suggested in v3)?

5. **Is there a minimum `p_available` below which the problem becomes degenerate?** At very low `p_available`, most patients can't be assigned to most slots, and the solver's output is almost entirely determined by availability rather than scores. The model has little to learn. Is `p_available = 0.5` a reasonable lower bound for the sweep?
