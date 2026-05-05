"""
Multi-slot scheduling solvers.

assign_slots                 — PyTorch (torch Tensor in/out); unconstrained greedy
                               used in evaluate.py when availability is None.
solve_multislot_availability — numpy (ndarray in/out); Gurobi ILP with availability
                               constraints. Used in evaluate.py, train_M2/M3/M4
                               val criterion, and M4 DFL gradient loop.
make_K_list                  — helper to convert capacity fractions to int counts.
"""
import numpy as np
import torch


def make_K_list(N: int, K_frac_list: list[float]) -> list[int]:
    """Convert per-slot capacity fractions to integer counts."""
    return [max(1, int(frac * N)) for frac in K_frac_list]


def assign_slots(scores: torch.Tensor, K_list: list[int]) -> torch.Tensor:
    """
    Unconstrained multi-slot assignment (PyTorch, no availability).

    Sort patients by triage score descending, fill slot 0 with the top K_0
    patients, slot 1 with the next K_1, etc.

    Args:
        scores: Tensor (N,) — higher = more urgent to schedule early
        K_list: list[int] — integer capacity per slot

    Returns:
        z: Tensor (N, T) — z[i, t] = 1 iff patient i assigned to slot t
    """
    N = scores.shape[0]
    T = len(K_list)

    idx = torch.argsort(scores, descending=True)
    z   = torch.zeros(N, T, device=scores.device, dtype=scores.dtype)

    start = 0
    for t, K_t in enumerate(K_list):
        K_t = int(K_t)
        if K_t <= 0:
            continue
        end      = min(start + K_t, N)
        z[idx[start:end], t] = 1.0
        start = end
        if start >= N:
            break

    return z


def solve_multislot_availability(
    scores: np.ndarray,
    K_list: list[int],
    availability: np.ndarray,
    delay: list[float] = (1.0, 3.0, 8.0),
    d_miss: float = 15.0,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Optimal multi-slot assignment with availability constraints (Gurobi ILP).

    Minimises the scheduling cost using triage scores as urgency weights:
      C = Σ_i Σ_t z[i,t]·(s_i·delay[t]+β) + Σ_i(1−Σ_t z[i,t])·s_i·d_miss

    Dropping the constant Σ_i s_i·d_miss, this reduces to:
      min  Σ_i Σ_t z[i,t]·(s_i·(delay[t]−d_miss)+β)

    The constraint matrix is totally unimodular (assignment + capacity), so
    the LP relaxation gives integer solutions; we use binary variables for
    explicitness.

    Args:
        scores:       (N,) float array — higher score = more urgent
        K_list:       list of T ints, capacity per slot
        availability: (N, T) int/bool array; 1 = patient i available for slot t
        delay:        length-T per-slot delay costs  (default [1.0, 3.0, 8.0])
        d_miss:       miss-penalty multiplier        (default 15.0)
        beta:         per-referral cost              (default 0.5)

    Returns:
        z: (N, T) int array; z[i, t] = 1 iff patient i assigned to slot t
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        raise ImportError(
            "gurobipy is required for the scheduling solver. "
            "Install with: pip install gurobipy"
        ) from e

    N, T  = len(scores), len(K_list)
    delay = [float(d) for d in delay]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.start()

    m = gp.Model(env=env)

    # Binary assignment variables: z[i, t] ∈ {0, 1}
    z = m.addVars(N, T, vtype=GRB.BINARY, name="z")

    # Fix unavailable patient-slot pairs (tighter than equality constraints)
    for i in range(N):
        for t in range(T):
            if not availability[i, t]:
                z[i, t].UB = 0.0

    # Slot capacity: at most K[t] patients per slot
    for t in range(T):
        m.addConstr(gp.quicksum(z[i, t] for i in range(N)) <= K_list[t])

    # Exclusivity: each patient assigned to at most one slot
    for i in range(N):
        m.addConstr(gp.quicksum(z[i, t] for t in range(T)) <= 1)

    # Objective: min Σ_{i,t} z[i,t] · (s_i·(delay[t]−d_miss) + β)
    m.setObjective(
        gp.quicksum(
            z[i, t] * (float(scores[i]) * (delay[t] - d_miss) + beta)
            for i in range(N) for t in range(T)
        ),
        GRB.MINIMIZE,
    )

    m.optimize()

    z_np = np.zeros((N, T), dtype=int)
    if m.Status == GRB.OPTIMAL:
        for i in range(N):
            for t in range(T):
                if z[i, t].X > 0.5:
                    z_np[i, t] = 1

    m.dispose()
    env.dispose()

    return z_np
