# Running Instructions: Availability Matrix and Model Training

All commands are run from the **project root** (`glaucoma-dfl-scheduling/`).

---

## Prerequisites

- Manifest CSV exists at `/data/lizhiwei/dfl_v2/manifest.csv`
  (produced by `src/catalog/data_pipeline_v2.py`)
- Old M1 checkpoint from the v3 run is present:
  `/data/lizhiwei/dfl_v2/models/M1_v3_seed42.pt`
- Old M1 prediction CSV is present (for evaluation):
  `/data/lizhiwei/dfl_v2/results/M1_v3_seed42.csv`
  (or any CSV with columns `patient_id`, `triage_score`, `true_severity`)

---

## Step 0 — Rename old M1 checkpoint

The new code expects `M1_seed{seed}.pt` (no `_v3` suffix).
Run once per seed:

```bash
mkdir -p /data/lizhiwei/dfl_v2/models

cp /data/lizhiwei/dfl_v2/models_v3/M1_v3_seed42.pt \
   /data/lizhiwei/dfl_v2/models/M1_seed42.pt

# Repeat for other seeds if needed
cp /data/lizhiwei/dfl_v2/models_v3/M1_v3_seed43.pt \
   /data/lizhiwei/dfl_v2/models/M1_seed43.pt

cp /data/lizhiwei/dfl_v2/models_v3/M1_v3_seed44.pt \
   /data/lizhiwei/dfl_v2/models/M1_seed44.pt
```

---

## Step 1 — Generate availability matrices (run once)

This creates two fixed `.npy` files used by all models for fair comparison:
- `data/availability/val_availability_seed100.npy`  — used during training (val decision cost)
- `data/availability/test_availability_seed200.npy` — used during final evaluation

```bash
python src/generate_availability.py \
    --manifest /data/lizhiwei/dfl_v2/manifest.csv
```

Default arguments match `config.py` (`p_available=0.7`, `T=3`, `seed-val=100`, `seed-test=200`).
Only needs to be run **once** — all subsequent training and evaluation reads the same files.

---

## Step 2 — Evaluate the old M1

Use the v4 evaluate.py with the test availability matrix so M1's metrics are on the
same constrained-assignment footing as M2b and M3.

```bash
python src/evaluate.py \
    --predictions /data/lizhiwei/dfl_v2/results/M1_v3_seed42.csv \
    --availability data/availability/test_availability_seed200.npy \
    --alpha 0 1 3 6 10 \
    --beta 0.5 \
    --delay 1.0 3.0 8.0 \
    --d_miss 15.0 \
    --K_frac_list 0.05 0.10 0.20 \
    --output /data/lizhiwei/dfl_v2/results/M1_seed42_metrics.json
```

Repeat for seeds 43 and 44.

> **Note:** `--K_frac_list 0.05 0.10 0.20` and the other cost parameters must
> match `config.py` so metrics are comparable across methods.

---

## Step 3 — Train M2b

M2b requires the M1 checkpoint (Step 0) and availability matrices (Step 1).
Training automatically saves:
- `models/M2b_seed42.pt` — best checkpoint (lowest val decision cost)
- `results/M2b_seed42.csv` — test-set predictions
- `results/M2b_seed42_metrics.json` — v4 scheduling metrics (availability-constrained)

```bash
python src/train_M2b.py --seed 42
python src/train_M2b.py --seed 43
python src/train_M2b.py --seed 44
```

**Smoke test** (a few epochs, verifies the pipeline end-to-end):

```bash
python src/train_M2b.py --seed 42 --smoke-test
```

---

## Step 4 — Train M3

M3 requires the M1 checkpoint (Step 0) and availability matrices (Step 1).
It re-runs Stage 2 (identical to M2b) then adds Stage 3 DFL fine-tuning.
Training automatically saves:
- `models/M3_stage2_seed42.pt` — best Stage-2 checkpoint
- `models/M3_seed42.pt` — best Stage-3 checkpoint (falls back to Stage 2 if DFL doesn't improve)
- `results/M3_seed42.csv` — test-set predictions
- `results/M3_seed42_metrics.json` — v4 scheduling metrics (availability-constrained)

```bash
python src/train_M3.py --seed 42
python src/train_M3.py --seed 43
python src/train_M3.py --seed 44
```

**Smoke test:**

```bash
python src/train_M3.py --seed 42 --smoke-test
```

---

## File dependency summary

```
src/generate_availability.py
    → data/availability/val_availability_seed100.npy   (read by train_M2b, train_M3)
    → data/availability/test_availability_seed200.npy  (read by evaluate.py, train_M2b, train_M3)

M1_seed{seed}.pt (renamed from M1_v3_seed{seed}.pt)
    → required by train_M2b.py and train_M3.py at startup

train_M2b.py / train_M3.py
    → produce {results,models}/M2b_seed{seed}.* and M3_seed{seed}.*
    → auto-evaluate with availability at end of run
```

---

## Notes

- All default paths (`--manifest`, `--output-dir`, `--model-dir`) point to
  `/data/lizhiwei/dfl_v2/` — override on the CLI if your paths differ.
- `data/availability/` is relative to the **project root** and is always read
  from there regardless of `--output-dir`.  If running from a different working
  directory, pass an absolute path via `--availability` to `evaluate.py`.
- To sweep severity-label fractions (Exp 2), add `--severity-fraction 0.25` etc.
