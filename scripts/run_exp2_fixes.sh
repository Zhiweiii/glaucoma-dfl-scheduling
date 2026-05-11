#!/usr/bin/env bash
# Exp 2 fixes:
#   Fix 1 — M3 re-run with n_val_avail=5 (stable checkpoint selection over
#            5 val availability realizations instead of 1)
#   Fix 2 — M4 Stage 3 sigma sweep {1.0, 2.0, 4.0} using existing Stage 2
#            checkpoints (--stage3-only), also with n_val_avail=5.
#            Rationale: alpha_hat std ≈ 1.2-1.5; sigma=0.5 (already run) was
#            too small (~0.35 std) to flip ILP assignments reliably.
#            Sweeping 1x, 2x, 4x the natural std to find a useful regime.
#
# Saves all results to exp2/ (M3 overwrites, M4 adds sigma-tagged files).
#
# Usage:
#   nohup bash scripts/run_exp2_fixes.sh \
#     > /data/lizhiwei/dfl_v2/v5/exp2/logs/fixes_master.log 2>&1 &

set -euo pipefail

AVAIL_DIR="/data/lizhiwei/dfl_v2/v5/availability_r5"
OUT_DIR="/data/lizhiwei/dfl_v2/v5/exp2/results"
MODEL_DIR="/data/lizhiwei/dfl_v2/v5/exp2/models"
LOG_DIR="/data/lizhiwei/dfl_v2/v5/exp2/logs"
PYTHON=".venv/bin/python"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/fixes_master.log"; }

# ── Fix 1: M3 with n_val_avail=5 ──────────────────────────────────────────────
for seed in 42 43 44; do
    log "Starting M3-fix seed${seed} (n_val_avail=5)..."
    $PYTHON src/train_M3.py \
        --seed        "$seed" \
        --avail-dir   "$AVAIL_DIR" \
        --output-dir  "$OUT_DIR" \
        --model-dir   "$MODEL_DIR" \
        --n-val-avail 5 \
        > "$LOG_DIR/M3fix_seed${seed}.log" 2>&1
    log "M3-fix seed${seed} done."
done

# ── Fix 2: M4 Stage 3 sigma sweep (stage3-only, n_val_avail=5) ────────────────
# sigma=0.5 already ran in exp2. Sweeping larger values (1x, 2x, 4x alpha_hat std).
for sigma in 1.0 2.0 4.0; do
    for seed in 42 43 44; do
        log "Starting M4 sigma=${sigma} seed=${seed}..."
        $PYTHON src/train_M4.py \
            --seed        "$seed" \
            --avail-dir   "$AVAIL_DIR" \
            --output-dir  "$OUT_DIR" \
            --model-dir   "$MODEL_DIR" \
            --n-val-avail 5 \
            --sigma       "$sigma" \
            --stage3-only \
            --stage2-ckpt "$MODEL_DIR/M4_stage2_seed${seed}.pt" \
            > "$LOG_DIR/M4_sigma${sigma}_seed${seed}.log" 2>&1
        log "M4 sigma=${sigma} seed${seed} done."
    done
done

log "=== Exp 2 fixes complete. Results in $OUT_DIR ==="
