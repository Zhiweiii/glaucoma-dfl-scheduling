import numpy as np


def simulate_availability(
    N: int,
    T: int,
    p_available: float | list[float] = 0.7,
    seed=None,
) -> np.ndarray:
    """
    Generate a binary (N, T) availability matrix.

    Args:
        N:           Number of patients
        T:           Number of slots
        p_available: Bernoulli probability per slot. Either a single float (same
                     probability for all slots) or a list of T floats (per-slot
                     probabilities). Per-slot values allow matching availability
                     to slot capacity so constraints are genuinely binding.
        seed:        Random seed.

    Returns:
        availability: (N, T) int array; availability[i, t] = 1 means patient i
                      can be assigned to slot t. Every patient is guaranteed at
                      least one available slot.
    """
    rng = np.random.RandomState(seed)

    if isinstance(p_available, (int, float)):
        p_per_slot = [float(p_available)] * T
    else:
        if len(p_available) != T:
            raise ValueError(f"p_available list length {len(p_available)} != T={T}")
        p_per_slot = [float(p) for p in p_available]

    availability = np.column_stack([
        rng.binomial(1, p, size=N) for p in p_per_slot
    ])

    # Guarantee each patient has at least one available slot.
    for i in range(N):
        if availability[i].sum() == 0:
            slot = rng.randint(T)
            availability[i, slot] = 1

    return availability
