#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon/Upgraded_dataset/s2_historical_bounds_native_v9
TABLE="$ROOT/s2_v9_recrop_progress.csv"
TARGET=/mnt/engg-niulab/yuyao/preprocessed_512/S2_6time_historical_bounds_native_v9
PRODUCT_CACHE=/diniuvol/yuyao/s2_v9_product_cache
CROP_CACHE=/diniuvol/yuyao/s2_v9_crop_cache
LOG="$ROOT/recrop.log"
STATUS="$ROOT/recrop.status"

mkdir -p "$ROOT" "$PRODUCT_CACHE" "$CROP_CACHE"
started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
started_epoch=$(date +%s)
printf 'running started_at=%s pid=%s\n' "$started_at" "$$" >"$STATUS"

set +e
PYTHONUNBUFFERED=1 /home/yuyao/miniconda3/envs/methane/bin/python \
  /home/yuyao/panopticon/Upgraded_dataset/s2_exact_point_recrop.py \
  --table "$TABLE" \
  --output-table "$TABLE" \
  --target-root "$TARGET" \
  --center-mode plume_bounds \
  --stac-dn-add-override 0 \
  --legacy-r20m-layout \
  --missing-source aws \
  --workers 24 \
  --aws-band-workers 2 \
  --aws-item-max-index 40 \
  --node-band-workers 4 \
  --node-global-workers 48 \
  --existing-cache-mode adaptive \
  --full-stage-min-tasks 24 \
  --product-scratch-dir "$PRODUCT_CACHE" \
  --crop-scratch-dir "$CROP_CACHE" \
  --sync-interval 250 \
  2>&1 | tee "$LOG"
exit_code=${PIPESTATUS[0]}
set -e

finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
finished_epoch=$(date +%s)
elapsed_seconds=$((finished_epoch - started_epoch))
printf 'finished exit_code=%s started_at=%s finished_at=%s elapsed_seconds=%s\n' \
  "$exit_code" "$started_at" "$finished_at" "$elapsed_seconds" >"$STATUS"
exit "$exit_code"
