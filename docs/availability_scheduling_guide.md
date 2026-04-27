# Availability-Constrained Multi-Slot Scheduling: Implementation Guide

## 1. Overview

This upgrade introduces **patient-slot availability constraints** into
the scheduling problem.

Key idea: - Cost function remains unchanged - Add feasibility
constraint: z\[i,t\] \<= availability\[i,t\] - This transforms the
problem from ranking → true scheduling

------------------------------------------------------------------------

## 2. Cost Function (UNCHANGED)

C(z, Y) = Σ\_{i,t} z\[i,t\] \* (alpha\[y_i\] \* delay\[t\] + beta) + Σ_i
(1 - Σ_t z\[i,t\]) \* alpha\[y_i\] \* d_miss

------------------------------------------------------------------------

## 3. New Constraint

availability\[i,t\] ∈ {0,1}

z\[i,t\] \<= availability\[i,t\]

Interpretation: - Patient i can only be assigned to slot t if available

------------------------------------------------------------------------

## 4. Simulating Availability

For all splits (train / val / test):

availability\[i,t\] \~ Bernoulli(p_available)

Recommended: p_available = 0.7

Ensure each patient has at least one available slot.

------------------------------------------------------------------------

## 5. Solver Update

Replace ranking solver with availability-aware greedy solver:

For each slot t: 1. Filter unassigned patients 2. Keep only those with
availability\[i,t\] = 1 3. Sort by predicted score 4. Assign top K_t

------------------------------------------------------------------------

## 6. Evaluation Pipeline

scores = model output availability = simulate_availability(...) z =
solve_multislot_availability(scores, K_list, availability) cost =
scheduling_cost(z, y)

Use SAME availability for all models.

------------------------------------------------------------------------

## 7. M3 Training Update

Inside each batch:

1.  Sample availability matrix
2.  For each perturbation:
    -   perturb scores
    -   solve with availability constraint
    -   compute cost
3.  Use same gradient estimator (with baseline)

------------------------------------------------------------------------

## 8. Recommended Settings

K_frac_list = \[0.05, 0.10, 0.15\] delay = \[1.0, 3.0, 8.0\] d_miss =
15.0 p_available = 0.7

------------------------------------------------------------------------

## 9. Key Insight

Original problem: → reduces to ranking

New problem: → constrained assignment

This makes the problem: ✔ non-separable ✔ decision-coupled ✔ truly
end-to-end

------------------------------------------------------------------------

## 10. One-Line Summary

Adding availability constraints turns the problem from a ranking problem
into a true scheduling problem under constraints.
