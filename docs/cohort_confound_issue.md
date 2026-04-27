# Cohort Confound Issue and Evaluation Fix

## 1. The Problem

### What we observed

M2b and M3 achieved near-perfect AUC-ROC (0.998 and 1.000) on the severity_test split.
This is implausibly high for a medical imaging task and indicates the model is not
learning what we intend.

### Root cause

The severity datasets (train, val, test) are constructed by mixing images from
**two completely separate clinical cohorts**:

| Label | Source | Cohort |
|-------|--------|--------|
| Grade 0 (no glaucoma) | LAC binary datasets (2-277G/J/C) | Diabetic screening program |
| Grade 1–4 (glaucoma) | Clinical records datasets (4-4116, 4-411G) | VF-tested glaucoma patients |

Confirmed by inspecting the manifest:

```
Grade-0 images  — binary_label set: 343 | severity_label set: 0
Sev 1–4 images  — binary_label set: 0   | severity_label set: 333
```

Every grade-0 image has a binary label (from the LAC screening cohort) and no
severity label. Every severity 1–4 image has a severity label (from the clinical
cohort) and no binary label. The two groups are **perfectly partitioned by data
source**.

### Why AUC → 1.0

AUC-ROC measures how well triage scores separate label=0 from label≥1.
Since the two cohorts likely differ systematically in imaging equipment, patient
demographics, image resolution, or preprocessing artifacts, the model trivially
learns **"which cohort does this image come from?"** rather than **"how severe is
the glaucoma?"**

This also inflates C_norm and recall@K, because grade-0 patients are ranked low
not due to any learned clinical knowledge but because of cohort-level visual
artifacts.

---

## 2. What is NOT affected

- **M1 training** — uses only binary_train/binary_val, both entirely within the
  LAC cohort. No confound.
- **Within-glaucoma severity ranking** — the pairwise accuracy among severity 1–4
  patients remains a valid measure. The model does learn something about glaucoma
  severity within the clinical cohort; the confound only makes grade-0 vs. grade≥1
  separation trivial.
- **The DFL mechanism itself** — the perturbation gradient, multi-slot solver, and
  cost structure are all correct. The confound is a data issue, not a modeling issue.

---

## 3. The Fix: Evaluate on Severity 1–4 Only

### Rationale

The core research question is:

> *Does DFL fine-tuning improve the ranking of glaucoma patients by severity,
> relative to severity CE alone?*

This question only involves severity 1–4 patients. Grade-0 patients should be
excluded from the primary evaluation because:

1. Their ranking is trivially correct due to cohort artifacts, not learned severity.
2. Including them inflates all metrics (AUC, C_norm, recall@K) in a way that does
   not reflect the model's true clinical utility.

### Implementation

Filter the prediction CSV to `true_severity >= 1` before computing metrics:

```python
df = pd.read_csv(predictions_csv)
df = df[df["true_severity"] >= 1].reset_index(drop=True)   # ← add this line
```

Apply in `evaluate.py` either as a flag (`--severity-only`) or as the default for
the v3 evaluation pipeline.

### What metrics mean under this fix

| Metric | Interpretation |
|--------|---------------|
| C_norm | Scheduling cost ratio vs. random, **among glaucoma patients only** |
| C_regret | Gap from oracle that knows true severity (1–4) |
| recall@K | Fraction of severe (Y≥3) glaucoma patients captured in any slot |
| pairwise_acc | Fraction of (i,j) glaucoma pairs correctly ranked by triage score |
| AUC-ROC | Ability to discriminate severe (Y≥3) from mild (Y=1,2) glaucoma |

AUC now measures within-glaucoma severity discrimination — a meaningful and
non-trivial clinical task.

---

## 4. Modeling Strategy for Next Version

### Option A: Severity-only evaluation (minimal change, recommended)

Keep the current training setup (grade-0 included in severity splits) but evaluate
on severity 1–4 only. This is honest because:

- Training on grade-0 teaches the model to rank non-glaucoma patients low, which is
  clinically useful even if the learning mechanism is partly confounded.
- The evaluation no longer rewards cohort-level discrimination.
- No architecture or data construction changes needed.

**Implementation:** add `--severity-only` flag to `evaluate.py` and set it as
default for all v3 training scripts.

### Option B: Train on severity 1–4 only (cleaner, more work)

Remove grade-0 from all severity splits. The severity head learns only within-
glaucoma ranking (grades 1–4). Grade-0 routing at test time falls to the M1 binary
head.

**Tradeoffs:**
- Pro: completely eliminates the confound from training
- Con: severity_train shrinks from 2,209 → ~1,091; severity head never sees class 0
- Con: combined scoring (binary head + severity head) requires a new triage score
  formula and additional design decisions
- Con: larger code changes; harder to compare fairly with M1

### Option C: Use same-cohort grade-0 images (ideal, requires catalog work)

Source grade-0 images from VF-confirmed normal cases within the same clinical
cohort as severity 1–4. This eliminates the confound at the data level.

**Tradeoffs:**
- Pro: the cleanest scientific fix; no evaluation workaround needed
- Con: requires checking whether same-cohort grade-0 cases exist in the eye-ai
  catalog and constructing new datasets (a significant data pipeline effort)
- Con: may not be feasible if the clinical cohort only enrolled glaucoma suspects

---

## 5. Recommended Immediate Action

**For the current v3 results:** re-evaluate all three models (M1, M2b, M3) with
severity 1–4 only to get honest comparative metrics.

```bash
uv run python src/evaluate.py \
    --predictions /data/lizhiwei/dfl_v2/results_v3/M1_v3_seed42.csv \
    --severity-only \
    --K_frac_list 0.05 0.10 0.20 \
    --output /data/lizhiwei/dfl_v2/results_v3/M1_v3_seed42_sev_only_metrics.json

uv run python src/evaluate.py \
    --predictions /data/lizhiwei/dfl_v2/results_v3/M2b_v3_seed42.csv \
    --severity-only \
    --K_frac_list 0.05 0.10 0.20 \
    --output /data/lizhiwei/dfl_v2/results_v3/M2b_v3_seed42_sev_only_metrics.json

uv run python src/evaluate.py \
    --predictions /data/lizhiwei/dfl_v2/results_v3/M3_v3_seed42.csv \
    --severity-only \
    --K_frac_list 0.05 0.10 0.20 \
    --output /data/lizhiwei/dfl_v2/results_v3/M3_v3_seed42_sev_only_metrics.json
```

**For the next modeling version:** adopt Option A (severity-only evaluation as
default) and acknowledge the cohort confound as a limitation in the report.
State that a future version would require same-cohort grade-0 images (Option C)
for a fully unconfounded evaluation.
