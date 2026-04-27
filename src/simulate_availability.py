import numpy as np


def simulate_availability(N: int, T: int, p_available: float = 0.7, seed=None) -> np.ndarray:
    """
    Generate a binary (N, T) availability matrix.

    Args:
        N:           Number of patients
        T:           Number of slots
        p_available: Bernoulli probability each patient is available per slot
        seed:        Random seed. Pass None during M3 training for stochastic sampling.

    Returns:
        availability: (N, T) int array; availability[i, t] = 1 means patient i can be
                      assigned to slot t.  Every patient is guaranteed at least one
                      available slot.
    """
    rng = np.random.RandomState(seed)
    availability = rng.binomial(1, p_available, size=(N, T))

    # Guarantee each patient has at least one available slot.
    for i in range(N):
        if availability[i].sum() == 0:
            slot = rng.randint(T)
            availability[i, slot] = 1

    return availability
