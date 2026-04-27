"""
Multi-slot scheduling solver (PyTorch).

assign_slots is used:
  - at test time by ALL methods (M1, M2, M3)
  - during training ONLY by M3 Stage 3 (inside the DFL loss)
"""
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
