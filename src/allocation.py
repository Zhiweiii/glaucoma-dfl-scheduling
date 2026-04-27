"""
Multi-slot scheduling solvers.

assign_slots                 — PyTorch (torch Tensor in/out); used in M3 Stage 3
                               backward pass.
solve_multislot_availability — numpy (ndarray in/out); used in evaluate.py, M2b/M3
                               val criterion, and M3 DFL gradient loop (after
                               .detach().cpu().numpy()).
"""
import numpy as np
import torch


def make_K_list(N: int, K_frac_list: list[float]) -> list[int]:
    """Convert per-slot capacity fractions to integer counts."""
    return [max(1, int(frac * N)) for frac in K_frac_list]


def assign_slots(scores: torch.Tensor, K_list: list[int]) -> torch.Tensor:
    """
    Multi-slot scheduling solver.

    Sort patients by triage score descending, fill slot 1 with the top K1
    patients, slot 2 with the next K2, etc.

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
) -> np.ndarray:
    """
    Greedy multi-slot assignment with availability constraints (numpy).

    Slots are processed in increasing index order (t=0 first, t=T-1 last).
    With delay=[1.0, 3.0, 8.0] this fills the lowest-delay slot first, matching
    the unconstrained assign_slots behaviour so the all-ones regression test holds.

    Args:
        scores:       (N,) float array — higher score = more urgent
        K_list:       list of T ints, capacity per slot
        availability: (N, T) int array; 1 = patient i may be assigned to slot t

    Returns:
        z: (N, T) int array; z[i, t] = 1 iff patient i assigned to slot t
    """
    N = len(scores)
    T = len(K_list)
    z        = np.zeros((N, T), dtype=int)
    assigned = np.zeros(N, dtype=bool)

    for t in range(T):
        eligible     = (~assigned) & (availability[:, t] == 1)
        eligible_idx = np.where(eligible)[0]
        if len(eligible_idx) == 0:
            continue
        sorted_eligible = eligible_idx[np.argsort(-scores[eligible_idx])]
        n_assign = min(K_list[t], len(sorted_eligible))
        for idx in sorted_eligible[:n_assign]:
            z[idx, t]  = 1
            assigned[idx] = True

    return z
