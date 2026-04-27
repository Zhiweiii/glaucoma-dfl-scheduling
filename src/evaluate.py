"""
Framework-agnostic evaluation: prediction CSV → scheduling metrics JSON.

Works identically for M1, M2b, M3, and the legacy Keras M1 — any model that
can write the prediction CSV format.

Problem: multi-slot scheduling.  Patients are assigned to one of T time slots
(or left unscheduled).  Cost is delay-sensitive:

  C(z, Y) = Σ_i Σ_t z_{i,t} (α_{Yi}·delay[t] + β)
           + Σ_i (1 − Σ_t z_{i,t}) α_{Yi}·d_miss

Input CSV columns:
    patient_id   — unique identifier (image_rid or subject id)
    triage_score — scalar ranking score (higher = more urgent to schedule early)
    true_severity — ground-truth severity label ∈ {0,1,2,3,4}

Output JSON keys:
    N, K_list, C_total, C_oracle, C_random, C_norm, C_regret,
    recall_at_K, pairwise_accuracy

Usage:
    python src/evaluate.py \\
        --predictions results/M1_seed42.csv \\
        --alpha 0 1 3 6 10 --beta 0.5 \\
        --delay 1.0 3.0 8.0 --d_miss 15.0 --K_frac_list 0.10 0.20 0.30 \\
        --output results/M1_seed42_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.allocation import solve_multislot_availability


# ── Allocation solver (numpy — unconstrained, evaluation only) ───────────────

def assign_slots(scores: np.ndarray, K_frac_list: list[float]) -> np.ndarray:
    """
    Greedy multi-slot assignment (numpy).

    Returns z: (N, T) binary matrix; z[i, t] = 1 iff patient i assigned to slot t.
    """
    N = len(scores)
    T = len(K_frac_list)
    z = np.zeros((N, T))
    sorted_idx = np.argsort(scores)[::-1]
    offset = 0
    for t, frac in enumerate(K_frac_list):
        Kt = min(max(1, int(frac * N)), N - offset)
        if Kt <= 0:
            break
        z[sorted_idx[offset : offset + Kt], t] = 1.0
        offset += Kt
    return z


# ── Cost functions ────────────────────────────────────────────────────────────

def scheduling_cost(
    z: np.ndarray,
    true_severity: np.ndarray,
    alpha: list[float],
    delay: list[float],
    beta: float,
    d_miss: float,
) -> float:
    """
    Multi-slot scheduling cost.

    C(z, Y) = Σ_i Σ_t z_{i,t} (α_{Yi}·delay[t] + β)
             + Σ_i (1 − Σ_t z_{i,t}) α_{Yi}·d_miss

    Args:
        z:             (N, T) binary assignment matrix
        true_severity: (N,)  integer severity labels
        alpha:         length-5 list of missed-referral costs per severity
        delay:         length-T list of delay weights per slot
        beta:          per-referral cost
        d_miss:        penalty multiplier for unscheduled patients
    """
    alpha_arr = np.array(alpha)
    delay_arr = np.array(delay)
    alpha_y   = alpha_arr[true_severity.astype(int)]  # (N,)
    assigned  = z.sum(axis=1)                          # (N,)

    assigned_cost   = float((z * (alpha_y[:, None] * delay_arr[None, :] + beta)).sum())
    unassigned      = (assigned == 0)
    unassigned_cost = float((alpha_y[unassigned] * d_miss).sum())
    return assigned_cost + unassigned_cost


def oracle_cost(
    true_severity: np.ndarray,
    alpha: list[float],
    delay: list[float],
    beta: float,
    K_frac_list: list[float],
    d_miss: float,
    availability: np.ndarray | None = None,
) -> float:
    """
    Lower bound: oracle knows true severity and assigns greedily by α_{Yi}.
    When availability is provided, uses the availability-constrained solver so
    the oracle is subject to the same feasibility constraints as the models.
    """
    oracle_scores = np.array(alpha)[true_severity.astype(int)].astype(float)
    N      = len(oracle_scores)
    K_list = [max(1, int(f * N)) for f in K_frac_list]
    if availability is not None:
        z_oracle = solve_multislot_availability(oracle_scores, K_list, availability)
    else:
        z_oracle = assign_slots(oracle_scores, K_frac_list)
    return scheduling_cost(z_oracle, true_severity, alpha, delay, beta, d_miss)


def random_cost(
    true_severity: np.ndarray,
    alpha: list[float],
    delay: list[float],
    beta: float,
    K_frac_list: list[float],
    d_miss: float,
    n_samples: int = 1000,
    seed: int = 0,
    availability: np.ndarray | None = None,
) -> float:
    """
    Expected cost under random multi-slot assignment (Monte Carlo).
    This is a reference quantity for normalisation, NOT a method.

    When availability is provided, each random sample also uses the
    availability-constrained solver so C_random is on the same feasibility
    footing as the models (making C_norm = C_total / C_random meaningful).
    """
    N      = len(true_severity)
    K_list = [min(max(1, int(f * N)), N) for f in K_frac_list]
    costs  = []

    if availability is not None:
        for s in range(n_samples):
            rng = np.random.RandomState(seed * 10000 + s)
            z   = solve_multislot_availability(rng.rand(N), K_list, availability)
            costs.append(scheduling_cost(z, true_severity, alpha, delay, beta, d_miss))
    else:
        rng = np.random.RandomState(seed)
        T   = len(K_frac_list)
        for _ in range(n_samples):
            perm   = rng.permutation(N)
            z      = np.zeros((N, T))
            offset = 0
            for t, Kt in enumerate(K_list):
                avail = min(Kt, N - offset)
                if avail <= 0:
                    break
                z[perm[offset : offset + avail], t] = 1.0
                offset += avail
            costs.append(scheduling_cost(z, true_severity, alpha, delay, beta, d_miss))

    return float(np.mean(costs))


# ── Supporting metrics ────────────────────────────────────────────────────────

def recall_at_K(
    z: np.ndarray,
    true_severity: np.ndarray,
    severe_threshold: int = 3,
) -> float:
    """Fraction of truly severe patients (Y ≥ threshold) captured across all slots."""
    assigned = z.sum(axis=1)  # (N,) — 1 if scheduled in any slot
    severe = true_severity >= severe_threshold
    if severe.sum() == 0:
        return float("nan")
    return float((assigned * severe).sum() / severe.sum())


def pairwise_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Fraction of discordant pairs (Y_i ≠ Y_j) where the model's ranking agrees:
    (Y_i > Y_j) ↔ (s_i > s_j).
    """
    s = scores.astype(float)
    y = labels.astype(float)
    label_diff = y[:, None] - y[None, :]
    score_diff = s[:, None] - s[None, :]
    upper   = np.triu(label_diff != 0, k=1)
    if upper.sum() == 0:
        return float("nan")
    correct = ((label_diff > 0) == (score_diff > 0)) & upper
    return float(correct.sum() / upper.sum())


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    predictions_csv: str | Path,
    alpha: list[float] | None = None,
    delay: list[float] | None = None,
    beta: float = 0.5,
    K_frac_list: list[float] | None = None,
    d_miss: float = 15.0,
    n_random: int = 1000,
    availability: np.ndarray | None = None,
    severity_only: bool = True,
) -> dict:
    """
    Compute all scheduling metrics from a prediction CSV.
    Identical logic for M1, M2b, M3, and the legacy Keras model.

    Args:
        predictions_csv: path to CSV with (patient_id, triage_score, true_severity)
        alpha:        missed-referral costs [α_0…α_4]     (default [0,1,3,6,10])
        delay:        per-slot delay weights               (default [1.0,3.0,8.0])
        beta:         per-referral cost                    (default 0.5)
        K_frac_list:  per-slot capacity fractions          (default [0.10,0.20,0.30])
        d_miss:       unscheduled patient penalty          (default 15.0)
        n_random:     MC samples for the random baseline
        availability: optional (N, T) int array.  When provided, model/oracle/random
                      all use the availability-constrained solver so C_norm is
                      comparable across methods.  Pass the same matrix to all models
                      being compared.  None → unconstrained (v3 behaviour).
                      When severity_only=True, this array must be pre-filtered to
                      the same severity 1–4 rows before being passed here.
        severity_only: if True (default), restrict evaluation to true_severity >= 1.
                      Excludes grade-0 patients whose ranking is trivially correct due
                      to cohort-level imaging differences (see docs/cohort_confound_issue.md).
                      AUC reports severe (Y≥3) vs mild (Y=1,2) discrimination.

    Returns dict with keys:
        N, K_list, C_total, C_oracle, C_random,
        C_norm   = C_total / C_random   (< 1 means better than random)
        C_regret = C_total − C_oracle   (gap from oracle)
        recall_at_K, pairwise_accuracy
    """
    if alpha is None:
        alpha = [0, 1, 3, 6, 10]
    if delay is None:
        delay = [1.0, 3.0, 8.0]
    if K_frac_list is None:
        K_frac_list = [0.10, 0.20, 0.30]

    df = pd.read_csv(predictions_csv)
    required = {"triage_score", "true_severity"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")

    if severity_only:
        df = df[df["true_severity"] >= 1].reset_index(drop=True)

    scores = df["triage_score"].values.astype(float)
    labels = df["true_severity"].values.astype(int)
    N      = len(scores)
    K_list = [max(1, int(np.floor(f * N))) for f in K_frac_list]

    if availability is not None and len(availability) != N:
        raise ValueError(
            f"availability has {len(availability)} rows but filtered predictions have {N}. "
            "When severity_only=True, pre-filter the availability matrix to the same "
            "severity 1–4 rows before passing it (see docs/cohort_confound_issue.md)."
        )

    if availability is not None:
        z_model = solve_multislot_availability(scores, K_list, availability)
    else:
        z_model = assign_slots(scores, K_frac_list)
    C_total  = scheduling_cost(z_model, labels, alpha, delay, beta, d_miss)
    C_oracle = oracle_cost(labels, alpha, delay, beta, K_frac_list, d_miss, availability)
    C_rand   = random_cost(labels, alpha, delay, beta, K_frac_list, d_miss, n_random,
                           availability=availability)

    if severity_only:
        # AUC: severe (Y≥3) vs mild (Y=1,2); grade-0 excluded above
        binary_labels = (labels >= 3).astype(int)
    else:
        binary_labels = (labels > 0).astype(int)
    n_pos = binary_labels.sum()
    auc = float(roc_auc_score(binary_labels, scores)) if 0 < n_pos < len(binary_labels) else float("nan")

    return {
        "N":                 N,
        "K_list":            K_list,
        "C_total":           float(C_total),
        "C_oracle":          float(C_oracle),
        "C_random":          float(C_rand),
        "C_norm":            float(C_total / max(C_rand, 1e-8)),
        "C_regret":          float(C_total - C_oracle),
        "recall_at_K":       recall_at_K(z_model, labels),
        "pairwise_accuracy": pairwise_accuracy(scores, labels),
        "auc_roc":           auc,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a prediction CSV against the multi-slot scheduling cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predictions", required=True,
                   help="CSV with columns: patient_id, triage_score, true_severity")
    p.add_argument("--alpha",       nargs=5, type=float, default=[0, 1, 3, 6, 10],
                   metavar=("a0", "a1", "a2", "a3", "a4"))
    p.add_argument("--delay",       nargs="+", type=float, default=[1.0, 3.0, 8.0])
    p.add_argument("--d_miss",      type=float, default=15.0)
    p.add_argument("--beta",        type=float, default=0.5)
    p.add_argument("--K_frac_list", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--n_random",    type=int,   default=1000)
    p.add_argument("--availability", default=None,
                   help="Path to .npy availability matrix (N, T). "
                        "If given, model/oracle/random all use availability-constrained solver. "
                        "When --severity-only, this matrix must be pre-filtered to severity 1–4 rows.")
    p.add_argument("--severity-only", dest="severity_only", action="store_true", default=True,
                   help="Evaluate on severity 1–4 only, excluding grade-0 patients "
                        "(recommended; see docs/cohort_confound_issue.md)")
    p.add_argument("--no-severity-only", dest="severity_only", action="store_false",
                   help="Include all severity levels (grade-0 through 4) in evaluation")
    p.add_argument("--output",      required=True, help="Output JSON path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    availability = np.load(args.availability) if args.availability else None
    metrics = evaluate(
        args.predictions,
        alpha=args.alpha,
        delay=args.delay,
        beta=args.beta,
        K_frac_list=args.K_frac_list,
        d_miss=args.d_miss,
        n_random=args.n_random,
        availability=availability,
        severity_only=args.severity_only,
    )
    print(json.dumps(metrics, indent=2))
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
