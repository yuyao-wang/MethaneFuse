#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon/Upgraded_dataset/s2_historical_bounds_native_v9
DATA_ROOT=/transferdiniu2/yuyao/final_crop/s2_historical_bounds_native_32_v9
TRAIN_CSV="$DATA_ROOT/train_patches_32.csv"
TEST_CSV="$DATA_ROOT/test_patches_32.csv"
CACHE=/diniuvol/yuyao/s2_v9_vit_cache
PYTHON=/home/yuyao/miniconda3/envs/methane/bin/python
PIPELINE=/home/yuyao/panopticon/Upgraded_dataset/s2_6time_cdse_legacy512_rebuild.py
DISTRIBUTION=/home/yuyao/panopticon/Upgraded_dataset/s2_patch_distribution_audit.py
LOG="$ROOT/crop32_audit.log"
STATUS="$ROOT/crop32_audit.status"

started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
started_epoch=$(date +%s)
printf 'running started_at=%s pid=%s\n' "$started_at" "$$" >"$STATUS"

set +e
{
  "$PYTHON" "$PIPELINE" audit \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    --audit-json "$ROOT/crop32_audit.json" \
    --path-audit-rows 5000 \
    --path-stat-workers 64

  "$PYTHON" "$DISTRIBUTION" \
    --cohort "train::$TRAIN_CSV::$CACHE" \
    --cohort "test::$TEST_CSV::$CACHE" \
    --path-columns path_t0,path_prev1,path_prev2,path_prev3,path_seasonal,path_year \
    --timepoints t0 prev1 prev2 prev3 seasonal year \
    --staged-maximum 3000 \
    --sample-maximum 2000 \
    --workers 64 \
    --output "$ROOT/crop32_distribution.json"
} 2>&1 | tee "$LOG"
exit_code=${PIPESTATUS[0]}
set -e

finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
finished_epoch=$(date +%s)
elapsed_seconds=$((finished_epoch - started_epoch))
printf 'finished exit_code=%s started_at=%s finished_at=%s elapsed_seconds=%s\n' \
  "$exit_code" "$started_at" "$finished_at" "$elapsed_seconds" >"$STATUS"
exit "$exit_code"
