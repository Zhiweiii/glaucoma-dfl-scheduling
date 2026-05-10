#!/usr/bin/env bash
# Exp 2: re-run M1-M4 against r5 availability (p=[0.25,0.50,1.00])
# M1 checkpoints are reused from exp1 (no retraining); M2/M3/M4 are retrained.
#
# Usage:
#   nohup bash scripts/run_exp2.sh > /data/lizhiwei/dfl_v2/v5/exp2/logs/master.log 2>&1 &

set -euo pipefail

AVAIL_DIR="/data/lizhiwei/dfl_v2/v5/availability_r5"
OUT_DIR="/data/lizhiwei/dfl_v2/v5/exp2/results"
MODEL_DIR="/data/lizhiwei/dfl_v2/v5/exp2/models"
LOG_DIR="/data/lizhiwei/dfl_v2/v5/exp2/logs"
EXP1_MODEL_DIR="/data/lizhiwei/dfl_v2/v5/exp1/models"
EXP1_RESULT_DIR="/data/lizhiwei/dfl_v2/v5/exp1/results"
PYTHON=".venv/bin/python"

mkdir -p "$OUT_DIR" "$MODEL_DIR" "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"; }

# ── Copy M1 checkpoints from exp1 (not retrained) ────────────────────────────
log "Copying M1 checkpoints from exp1..."
for seed in 42 43 44; do
    cp "$EXP1_MODEL_DIR/M1_seed${seed}.pt" "$MODEL_DIR/"
done
log "M1 checkpoints ready."

# ── Re-evaluate M1 with new availability ──────────────────────────────────────
log "Re-evaluating M1 with r5 availability..."
$PYTHON - <<'PYEOF'
import sys, json, shutil, numpy as np
sys.path.insert(0, '.')
from src.evaluate import evaluate
from config import CONFIG
import os

avail  = np.load(os.environ.get('AVAIL_DIR', '/data/lizhiwei/dfl_v2/v5/availability_r5') + '/test_availability_seed200.npy')
out    = os.environ.get('OUT_DIR',  '/data/lizhiwei/dfl_v2/v5/exp2/results')
exp1   = os.environ.get('EXP1_RESULT_DIR', '/data/lizhiwei/dfl_v2/v5/exp1/results')

for seed in [42, 43, 44]:
    src = f'{exp1}/M1_seed{seed}.csv'
    dst = f'{out}/M1_seed{seed}.csv'
    shutil.copy(src, dst)
    m = evaluate(dst, alpha=CONFIG['alpha'], beta=CONFIG['beta'],
                 K_frac_list=CONFIG['K_frac_list'], delay=CONFIG['delay'],
                 d_miss=CONFIG['d_miss'], availability=avail, severity_only=True)
    with open(f'{out}/M1_seed{seed}_metrics.json', 'w') as f:
        json.dump(m, f, indent=2)
    print(f'  M1 seed{seed}: C_norm={m["C_norm"]:.4f}  recall@K={m["recall_at_K"]:.4f}')
PYEOF
log "M1 evaluation done."

# ── Train M2 ──────────────────────────────────────────────────────────────────
for seed in 42 43 44; do
    log "Starting M2 seed${seed}..."
    $PYTHON src/train_M2.py \
        --seed "$seed" \
        --avail-dir  "$AVAIL_DIR" \
        --output-dir "$OUT_DIR" \
        --model-dir  "$MODEL_DIR" \
        > "$LOG_DIR/M2_seed${seed}.log" 2>&1
    log "M2 seed${seed} done."
done

# ── Train M3 ──────────────────────────────────────────────────────────────────
for seed in 42 43 44; do
    log "Starting M3 seed${seed}..."
    $PYTHON src/train_M3.py \
        --seed "$seed" \
        --avail-dir  "$AVAIL_DIR" \
        --output-dir "$OUT_DIR" \
        --model-dir  "$MODEL_DIR" \
        > "$LOG_DIR/M3_seed${seed}.log" 2>&1
    log "M3 seed${seed} done."
done

# ── Train M4 ──────────────────────────────────────────────────────────────────
for seed in 42 43 44; do
    log "Starting M4 seed${seed}..."
    $PYTHON src/train_M4.py \
        --seed "$seed" \
        --avail-dir  "$AVAIL_DIR" \
        --output-dir "$OUT_DIR" \
        --model-dir  "$MODEL_DIR" \
        > "$LOG_DIR/M4_seed${seed}.log" 2>&1
    log "M4 seed${seed} done."
done

log "=== Exp 2 complete. Results in $OUT_DIR ==="
