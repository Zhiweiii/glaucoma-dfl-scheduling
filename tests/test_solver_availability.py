import numpy as np
import torch

from src.allocation import assign_slots, solve_multislot_availability


def test_all_ones_matches_unconstrained():
    """With all-ones availability, constrained solver must match assign_slots exactly."""
    np.random.seed(0)
    N, T = 100, 3
    scores_np = np.random.rand(N)
    K_list    = [5, 10, 15]
    availability = np.ones((N, T), dtype=int)

    z_constrained   = solve_multislot_availability(scores_np, K_list, availability)
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

    z = solve_multislot_availability(scores, K_list, availability)
    assert z[0].sum() == 0, "Fully unavailable patient must not be assigned"


def test_total_assignments_within_capacity():
    """Total number of assignments must never exceed sum(K_list)."""
    np.random.seed(42)
    N, T   = 200, 3
    scores = np.random.rand(N)
    K_list = [10, 20, 30]
    availability = np.random.binomial(1, 0.5, size=(N, T))

    z = solve_multislot_availability(scores, K_list, availability)
    assert z.sum() <= sum(K_list)


def test_each_patient_assigned_at_most_once():
    """No patient may appear in more than one slot."""
    np.random.seed(7)
    N, T   = 100, 3
    scores = np.random.rand(N)
    K_list = [10, 10, 10]
    availability = np.random.binomial(1, 0.7, size=(N, T))

    z = solve_multislot_availability(scores, K_list, availability)
    assert (z.sum(axis=1) <= 1).all()


def test_availability_respected():
    """Assigned patients must only appear in slots where they are available."""
    np.random.seed(99)
    N, T   = 80, 3
    scores = np.random.rand(N)
    K_list = [8, 8, 8]
    availability = np.random.binomial(1, 0.6, size=(N, T))

    z = solve_multislot_availability(scores, K_list, availability)
    # For every patient-slot pair that is assigned, availability must be 1
    assert (z <= availability).all()
