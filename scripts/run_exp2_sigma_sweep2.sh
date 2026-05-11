#!/usr/bin/env bash
# Exp 2 sigma sweep continuation: σ ∈ {6.0, 8.0}
# Reuses Stage 2 checkpoints from exp2/models/ (--stage3-only).
#
# Usage:
#   nohup bash scripts/run_exp2_sigma_sweep2.sh \
#     > /data/lizhiwei/dfl_v2/v5/exp2/logs/sigma_sweep2_master.log 2>&1 &

set -euo pipefail

AVAIL_DIR="/data/lizhiwei/dfl_v2/v5/availability_r5"
OUT_DIR="/data/lizhiwei/dfl_v2/v5/exp2/results"
MODEL_DIR="/data/lizhiwei/dfl_v2/v5/exp2/models"
LOG_DIR="/data/lizhiwei/dfl_v2/v5/exp2/logs"
PYTHON=".venv/bin/python"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/sigma_sweep2_master.log"; }

for sigma in 6.0 8.0; do
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

log "=== Sigma sweep 2 complete ==="
