import numpy as np
import torch

from src.allocation import assign_slots, solve_multislot_availability

# Cost parameters used by all tests (match CONFIG defaults)
DELAY  = [1.0, 3.0, 8.0]
D_MISS = 15.0
BETA   = 0.5


def test_all_ones_matches_unconstrained():
    """With all-ones availability, Gurobi optimal must match assign_slots exactly.

    When every patient is available for every slot and scores are distinct,
    the optimal assignment is: sort patients by score descending, fill slot 0
    up to K_0, then slot 1 up to K_1, etc. — identical to assign_slots.
    """
    np.random.seed(0)
    N, T = 100, 3
    scores_np = np.random.rand(N)
    K_list    = [5, 10, 15]
    availability = np.ones((N, T), dtype=int)

    z_constrained   = solve_multislot_availability(
        scores_np, K_list, availability,
        delay=DELAY, d_miss=D_MISS, beta=BETA,
    )
    z_unconstrained = assign_slots(
        torch.tensor(scores_np, dtype=torch.float32), K_list
    ).numpy().astype(int)

    assert np.array_equal(z_unconstrained, z_constrained)


def test_unavailable_patient_not_assigned():
    """Patient with all-zero availability row must not appear in any slot."""
    N, T   = 50, 3
    scores = np.ones(N)
    K_list = [10, 10, 10]
    availability = np.ones((N, T), dtype=int)
    availability[0, :] = 0

    z = solve_multislot_availability(
        scores, K_list, availability,
        delay=DELAY, d_miss=D_MISS, beta=BETA,
    )
    assert z[0].sum() == 0, "Fully unavailable patient must not be assigned"


def test_total_assignments_within_capacity():
    """Total number of assignments must never exceed sum(K_list)."""
    np.random.seed(42)
    N, T   = 200, 3
    scores = np.random.rand(N)
    K_list = [10, 20, 30]
    availability = np.random.binomial(1, 0.5, size=(N, T))

    z = solve_multislot_availability(
        scores, K_list, availability,
        delay=DELAY, d_miss=D_MISS, beta=BETA,
    )
    assert z.sum() <= sum(K_list)


def test_each_patient_assigned_at_most_once():
    """No patient may appear in more than one slot."""
    np.random.seed(7)
    N, T   = 100, 3
    scores = np.random.rand(N)
    K_list = [10, 10, 10]
    availability = np.random.binomial(1, 0.7, size=(N, T))

    z = solve_multislot_availability(
        scores, K_list, availability,
        delay=DELAY, d_miss=D_MISS, beta=BETA,
    )
    assert (z.sum(axis=1) <= 1).all()


def test_availability_respected():
    """Assigned patients must only appear in slots where they are available."""
    np.random.seed(99)
    N, T   = 80, 3
    scores = np.random.rand(N)
    K_list = [8, 8, 8]
    availability = np.random.binomial(1, 0.6, size=(N, T))

    z = solve_multislot_availability(
        scores, K_list, availability,
        delay=DELAY, d_miss=D_MISS, beta=BETA,
    )
    assert (z <= availability).all()


def test_optimal_beats_greedy_on_conflict():
    """Gurobi optimal must not be worse than the greedy on a hand-crafted conflict.

    Setup: 2 patients, 2 slots with capacity 1 each.
    - Patient 0: high score (9), only available for slot 1.
    - Patient 1: medium score (5), available for both slots.

    Greedy (fill slot 0 first): assigns patient 1 to slot 0, patient 0 to slot 1.
    Optimal: same — patient 0 gets slot 1 (their only option), patient 1 gets slot 0.
    Both produce the same assignment here; the test verifies correctness, not
    superiority, in this specific case.
    """
    scores       = np.array([9.0, 5.0])
    K_list       = [1, 1]
    availability = np.array([[0, 1],   # patient 0: only slot 1
                             [1, 1]])  # patient 1: both slots

    z = solve_multislot_availability(
        scores, K_list, availability,
        delay=DELAY, d_miss=D_MISS, beta=BETA,
    )
    assert z[0, 0] == 0, "Patient 0 must not be in slot 0 (unavailable)"
    assert z[0, 1] == 1, "Patient 0 should be assigned to slot 1"
    assert z[1, 0] == 1, "Patient 1 should be assigned to slot 0"
    assert z[1, 1] == 0, "Patient 1 must not be in both slots"
