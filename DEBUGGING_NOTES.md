# M3 DFL Debugging Notes

## Summary

Three bugs were identified that caused M3's DFL Stage 3 gradient to be nearly zero, 
explaining why M3 performed no better than M2b despite the decision-focused training.

---

## Bug 0 — M2b/M3 trunk frozen during Phase 1 (FIXED in commit f6b872a)

**Location:** `src/train_M2b.py` Phase 1, `src/train_M3.py` Stage 2 Phase 1

**Problem:** Phase 1 only trained `severity_head`, leaving the trunk frozen.
- M2b trunk received 10 total severity epochs (Phase 2 only)
- M2a trunk received 30 total severity epochs (20 Phase 1 + 10 Phase 2)
This explained why M2a beat M2b despite no binary warm-start.

**Fix (Option A):** Phase 1 now trains `trunk + severity_head` (backbone still frozen).
Same setup as M2a — trunk gets 30 total severity epochs for both M2b and M3.

**Status:** Fixed. `model.trunk.eval()` removed from loop; optimizer updated.

---

## Bug 3 — val_decision_cost includes grade-0 patients (FIXED in commit f6b872a)

**Location:** `val_decision_cost` in M2a, M2b, M3

**Problem:** Val cost was computed on all 548 severity_val patients (including
~274 grade-0), giving K_list=[27,54,109]. Test uses 333 severity-only patients
with K_list=[16,33,66]. Checkpoint selection optimised the wrong problem.

**Fix:** Filter to `sev_labels >= 1` inside `val_decision_cost`, and pass
`val_availability_sev = val_availability[val_sev_mask]` (severity-only rows).
Val K_list is now ~[13,27,55] — aligned with test [16,33,66].

**Status:** Fixed in M2a, M2b, M3.

---

## Bug 1 — sigma too small (FIXED)

**Location:** `config.py`

```python
"sigma": 0.5,   # was 0.1; perturbation noise std for randomised smoothing
```

**Impact (CRITICAL):** Severity scores are in [0, 10]. With sigma=0.1, perturbations 
rarely flip slot assignments when score gaps exceed 0.2. The REINFORCE gradient estimator 
sees nearly identical cost across Monte Carlo samples → gradient ≈ 0.

**Fix:** Changed `sigma` from `0.1` to `0.5`.

**Status:** Fixed.

---

## Bug 2 — DFL batch K_list mismatch (FIXED in commit d1cad66)

**Location:** `config.py` + `src/train_M3.py`

**Problem:** Stage 3 DFL loop used the same `train_loader` (batch_size=32).
With N=32 patients per batch:
- Training K_list = [1, 3, 6]   (5/10/20% of 32)
- Test     K_list = [16, 33, 66] (5/10/20% of 333 severity-only patients)

The DFL gradient was estimated for a trivially small scheduling problem (1–6 slots) 
while the test metric evaluated a full-sized problem (16–66 slots). 
The model was never trained on anything resembling the actual deployment setting.

**Fix applied:**
- Added `"batch_size_stage3": 256` to `config.py`
- Added separate `dfl_loader` in `src/train_M3.py` using `batch_size_stage3`
- Stage 3 loop now iterates `dfl_loader` instead of `train_loader`

With batch_size=256: K_list = [12, 25, 51] — much closer to test [16, 33, 66].

**Commit:** `d1cad66` — *M3 Stage 3 batch size fix*

---

## Bug 3 — val_decision_cost includes grade-0 patients (NOT YET FIXED)

**Location:** `src/train_M3.py`, function `val_decision_cost`

**Problem:** During Stage 3, checkpoint selection uses `val_decision_cost` computed 
on **all** 548 val patients (including 215 grade-0 patients with no clinical urgency). 
The test metric uses **only** 333 severity-only patients.

The model is checkpointed to minimize cost on a different patient mix than what is 
actually evaluated at test time.

**Fix:** Filter val loader to severity >= 1 only (same as `severity_only=True` used at test time).

**Status:** Pending.

---

## Evaluation consistency fixes (commit 6b0c3d5)

Several models were calling `evaluate()` without availability constraints or 
`severity_only=True`. Fixed in:

| File | Fix |
|------|-----|
| `src/train_M1.py` | Load `test_availability`, pass `availability=test_availability[sev_mask]`, `severity_only=True` |
| `src/train_M2a.py` | Same final evaluate() fix; also fixed `val_decision_cost` to use `solve_multislot_availability` |
| `src/dataset_construction/data_pipeline_v2.py` | Added binary_test dataset RID `5-ZMGJ` |

---

## Logged results

### Original M3 training (seed=42, batch_size=32 for Stage 3)
- **Log:** `/data/lizhiwei/dfl_v2/results/M3_seed42_train.log`
- **Metrics:** `/data/lizhiwei/dfl_v2/results/M3_seed42_metrics.json`
- **Model:** `/data/lizhiwei/dfl_v2/models/M3_seed42.pt`
- **Predictions:** `/data/lizhiwei/dfl_v2/results/M3_seed42.csv`

### Retrained M3 (seed=42, batch_size_stage3=256 — Bug 2 fix only)
- **Log:** `/data/lizhiwei/dfl_v2/results_bs256/M3_seed42_train.log`
- **Metrics:** `/data/lizhiwei/dfl_v2/results_bs256/M3_seed42_metrics.json`
- **Model:** `/data/lizhiwei/dfl_v2/models_bs256/M3_seed42.pt`
- **Predictions:** `/data/lizhiwei/dfl_v2/results_bs256/M3_seed42.csv`

### Other models (seed=42)
- **M1 metrics:** `/data/lizhiwei/dfl_v2/results/M1_seed42_metrics.json`
- **M2a metrics:** `/data/lizhiwei/dfl_v2/results/M2a_seed42_metrics.json`
- **M2b metrics:** `/data/lizhiwei/dfl_v2/results/M2b_seed42_metrics.json`

---

---

## Bug 4 — Training availability was stochastic per batch (FIXED)

**Location:** `src/train_M3.py` `dfl_step`, `src/generate_availability.py`, `config.py`

**Problem:** `dfl_step` called `simulate_availability(seed=None)` fresh every batch,
so M3 trained against a different availability realization each step. This is conceptually
wrong — the training cohort is fixed, so patient availability should be fixed. It also
misaligned training with evaluation: val/test use fixed pre-generated matrices, but
training used random ones. A further fairness problem: availability was not the same
across all model comparisons.

**Fix:**
- Added `"availability_seed_train": 0` to `config.py`
- Extended `generate_availability.py` to also generate `train_availability_seed0.npy`
  for the `severity_train` split (must re-run `generate_availability.py`)
- `GlaucomaDataset.__getitem__` now returns `idx` as a 5th element
- `dfl_step` takes `patient_idx` and `train_availability` parameters; indexes the
  pre-fixed matrix by `patient_idx[mask]` instead of simulating
- All DataLoader loops across M1/M2a/M2b/M3/eval_binary updated to unpack 5-tuples
- Removed unused `simulate_availability` and `assign_slots` imports from M2b/M3

**Status:** Fixed. Re-run `generate_availability.py` before next training run.

---

## Recommended next steps

1. Re-run `generate_availability.py` to generate `train_availability_seed0.npy`.
2. Retrain all models (M1 → M2a/M2b → M3) with all fixes applied (sigma=0.5, fixed train avail).
3. Run seeds 42, 43, 44 for full comparison.
