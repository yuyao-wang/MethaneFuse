#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon/Upgraded_dataset/s2_historical_bounds_native_v9
DATA_ROOT=/transferdiniu2/yuyao/final_crop/s2_historical_bounds_native_32_v9
TRAIN_CSV="$DATA_ROOT/train_patches_32.csv"
TEST_CSV="$DATA_ROOT/test_patches_32.csv"
CACHE=/diniuvol/yuyao/s2_v9_vit_cache
PYTHON=/home/yuyao/miniconda3/envs/panopticon/bin/python
SCRIPT=/home/yuyao/panopticon/Upgraded_dataset/s2_resnet18_diagnostic.py
STATUS="$ROOT/resnet_gate.status"

mkdir -p "$ROOT" "$CACHE"
started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
started_epoch=$(date +%s)
printf 'running started_at=%s pid=%s\n' "$started_at" "$$" >"$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PYTHON" "$SCRIPT" \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --mode current \
  --timepoints t0,prev1,prev2,prev3,seasonal,year \
  --band-indices 0,1,2,3,4,5,6,7,10,11 \
  --center-box 256 \
  --max-train-samples 20000 \
  --max-test-samples 10000 \
  --batch-size 512 \
  --epochs 3 \
  --num-workers 16 \
  --local-cache-dir "$CACHE" \
  --cache-workers 64 \
  --device cuda:0 \
  --output "$ROOT/resnet_gate_all6.json" \
  2>&1 | tee "$ROOT/resnet_gate_all6.log"
all6_exit=${PIPESTATUS[0]}

t0_exit=1
if [[ "$all6_exit" -eq 0 ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON" "$SCRIPT" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    --mode current \
    --timepoints t0 \
    --band-indices 0,1,2,3,4,5,6,7,10,11 \
    --center-box 256 \
    --max-train-samples 20000 \
    --max-test-samples 10000 \
    --batch-size 512 \
    --epochs 3 \
    --num-workers 16 \
    --local-cache-dir "$CACHE" \
    --cache-workers 64 \
    --device cuda:0 \
    --output "$ROOT/resnet_gate_t0.json" \
    2>&1 | tee "$ROOT/resnet_gate_t0.log"
  t0_exit=${PIPESTATUS[0]}
fi
set -e

finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
finished_epoch=$(date +%s)
elapsed_seconds=$((finished_epoch - started_epoch))
printf 'finished all6_exit=%s t0_exit=%s started_at=%s finished_at=%s elapsed_seconds=%s\n' \
  "$all6_exit" "$t0_exit" "$started_at" "$finished_at" "$elapsed_seconds" >"$STATUS"
if [[ "$all6_exit" -ne 0 ]]; then
  exit "$all6_exit"
fi
exit "$t0_exit"
