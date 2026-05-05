# V5 Project Design: Feature-Freezing Decomposition for Glaucoma Triage DFL

## 1. Motivation

The professor's feedback identified two problems with the current project structure:

1. **Narrative problem:** The scarce-severity-label vs. plentiful-binary-label story dominates the report but is not where the DFL contribution lives. It should be setup, not the main story.
2. **Data problem:** Grade-0 images come from a different clinical cohort (LAC diabetic screening) than grades 1–4 (VF-tested glaucoma patients). Models learn to distinguish cohorts, not severity. This was documented in `cohort_confound_issue.md`.

The professor suggested a single design change that resolves both problems: **freeze a trunk pre-trained on binary labels, then compare decision-blind vs. light-touch vs. DFL only on the downstream severity learner.**

This document describes the revised project design. The companion file `v5_implementation_spec.md` contains the exact implementation for Claude Code.

---

## 2. Two-Phase Architecture

### Phase 1: Pre-train on binary labels (done once, not the research contribution)

- Train a neural network (VGG-19 backbone + trunk) on the plentiful binary dataset (glaucoma vs. no glaucoma).
- This is the existing M1 training — no changes needed.
- The binary dataset is entirely within the LAC cohort, so there is no confound.

### Phase 2: Train a downstream severity learner (this is the DFL experiment)

- **Freeze the backbone and trunk** from Phase 1. Treat them as a fixed feature extractor.
- Attach a lightweight severity head on top of the frozen features.
- Train this head **only on severity-labeled patients (grades 1–4)** from the clinical cohort.
- Compare three training approaches for this head:
  - **M2 (baseline):** Train with cross-entropy on severity labels. Tune hyperparameters by validation CE loss.
  - **M3 (light-touch):** Train with cross-entropy on severity labels. Tune hyperparameters by validation scheduling cost.
  - **M4 (DFL):** Train by directly minimizing scheduling cost via perturbation gradients.

### Why this eliminates the cohort confound

Grade-0 images are never used in Phase 2 — not in training, not in validation, not in evaluation. The entire Phase 2 pipeline operates within the clinical cohort (grades 1–4 only). There is no opportunity for the model to learn cohort-distinguishing artifacts because only one cohort is present.

The frozen trunk may have learned some cohort-distinguishing features during Phase 1, but that is irrelevant: all three methods (M2a, M2b, M3) use the same frozen trunk. Any cohort-related features in the trunk affect all methods equally and wash out in the comparison.

### Why this simplifies the report

The binary classification step is mentioned once in the setup: "We pre-train a feature extractor on plentiful binary labels and freeze it." The rest of the report focuses exclusively on the Phase 2 comparison, which is where the DFL methodology applies.

---

## 3. Decision Problem: Availability-Constrained Multi-Slot Scheduling

The downstream decision problem is unchanged from v4: assign patients to limited treatment slots to minimize total scheduling cost.

### Formulation

- **Patients:** N patients with true severity grades Y_i ∈ {1, 2, 3, 4}.
- **Slots:** T = 3 time slots with capacities K_1, K_2, K_3 (set as fractions of N: 5%, 10%, 20%).
- **Delay costs:** delay = [1.0, 3.0, 8.0] — later slots incur higher delay.
- **Severity weights:** alpha[y] increases with severity (more severe patients are costlier to delay).
- **Availability:** Each patient has a binary availability vector; z[i,t] = 1 only if patient i is available for slot t.
- **Miss penalty:** Unassigned patients incur cost alpha[y_i] × d_miss.

### Solver

Greedy assignment: process slots in order (t = 0 first), assign highest-scoring available patients up to capacity.

This is an approximation. The report should state this explicitly and note that the greedy solver is the same across all methods, so any suboptimality affects all methods equally.

### Why this is a genuine scheduling problem (not just ranking)

Without availability constraints, optimal assignment reduces to sorting by score — a pure ranking problem where DFL has limited advantage over a good predictor. Availability constraints make the problem non-separable: the best assignment for patient i depends on which other patients are available for the same slots. This is the setting where DFL should shine.

---

## 4. What Varies Across Methods (The Comparison Table)

| | M2 (baseline) | M3 (light-touch) | M4 (DFL) |
|---|---|---|---|
| **Frozen layers** | Backbone + trunk | Backbone + trunk | Backbone + trunk |
| **Trainable parameters** | Severity head only | Severity head only | Severity head only |
| **Training loss** | Cross-entropy on severity labels | Cross-entropy on severity labels | Scheduling cost via perturbation gradients |
| **Hyperparameter tuning** | Validation CE loss | Validation scheduling cost | Validation scheduling cost |
| **What makes it decision-aware** | Nothing | Hyperparameter selection only | Training objective itself |

This table should appear in the methodology section of the report. It makes the experimental design immediately clear.

---

## 5. Evaluation

### Population

All evaluation is on severity 1–4 patients only. Grade-0 is excluded by design (they are not in the Phase 2 pipeline at all).

### Metrics

| Metric | What it measures |
|---|---|
| C_norm | Scheduling cost ratio vs. random baseline (lower is better) |
| C_regret | Gap from oracle with perfect severity knowledge |
| recall@K | Fraction of severe (grade ≥ 3) patients captured in any slot |
| pairwise_acc | Fraction of patient pairs correctly ranked by triage score |
| AUC-ROC | Discrimination between severe (grade ≥ 3) and mild (grade 1–2) |

### Baselines

Both the random baseline and the oracle must use the same availability matrix as the models. This ensures C_norm is a fair ratio.

### Interpretation guidance

- If M4 ≈ M3 ≈ M2: The frozen features are already so good that the downstream head cannot improve much. DFL adds no value in this regime. Discuss why (e.g., the scheduling problem may be nearly linear in scores, so decision-blind prediction suffices per Corollary 4.1 in the course notes).
- If M3 > M2 but M4 ≈ M3: Light-touch tuning captures most of the decision-aware benefit. The full DFL gradient is not worth the computational cost.
- If M4 > M3 > M2: Full DFL training provides genuine improvement. Discuss what the DFL gradient is learning that CE cannot.
- If M4 < M3: DFL may be overfitting to the training scheduling cost with scarce labels, or the perturbation gradient variance is too high. Diagnose by checking training vs. validation cost curves.

---

## 6. Story for the Report

1. **Introduction:** Background on glaucoma triage, the clinical need for severity-based scheduling.
2. **Problem Setup:** Two data sources (binary labels, severity labels). Feature-freezing decomposition. Present once, with a pipeline diagram, then move on.
3. **Methodology:**
   - 3.1 The scheduling problem (formulation, solver, availability constraints).
   - 3.2 The three methods (the comparison table above).
   - 3.3 Connection to course concepts: plug-in policies, solution maps, light-touch tuning (Ch. 3), perturbation gradients as randomized smoothing (Ch. 10).
4. **Experiments:** Results with interpretation of each metric. Ablation: with vs. without availability constraints.
5. **Discussion:** Diagnosis of when/why DFL helps or doesn't. Limitations (greedy solver, simulated availability, single cohort for severity).
6. **Conclusion.**
