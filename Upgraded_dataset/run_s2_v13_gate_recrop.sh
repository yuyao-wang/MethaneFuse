#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon/Upgraded_dataset/s2_historical_point_center_v13_gate
SOURCE=/home/yuyao/panopticon/Upgraded_dataset/s2_historical_point_center_v11/s2_v11_test_512.csv
TABLE="$ROOT/test_3tp_recrop.csv"
TARGET=/diniuvol/yuyao/s2_point_center_plus1000_v13_gate_512
PRODUCT_CACHE=/diniuvol/yuyao/s2_v13_gate_cache/products
CROP_CACHE=/diniuvol/yuyao/s2_v13_gate_cache/crops
LOG="$ROOT/test_3tp_recrop.log"
STATUS="$ROOT/test_3tp_recrop.status"
PYTHON=/home/yuyao/miniconda3/envs/panopticon/bin/python
PIPELINE=/home/yuyao/panopticon/Upgraded_dataset/s2_exact_point_recrop.py
LEGACY_CONFIG=/home/yuyao/methane_train/data_preprocess/configs/carbon_mapper_sentinel2_plume_download.yaml
EXISTING_PRODUCT_ROOTS="/diniuvol/yuyao/s2_cdse_point_repair_cache,/diniuvol/yuyao/s2_early_boundary_products,/mnt/engg-niulab/yuyao/sensors_raw_data/S2/raw_data_dir_s2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2_90360,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7"

mkdir -p "$ROOT" "$TARGET" "$PRODUCT_CACHE" "$CROP_CACHE"
if [[ ! -s "$TABLE" ]]; then
  cp "$SOURCE" "$TABLE"
fi

started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
started_epoch=$(date +%s)
printf 'running started_at=%s pid=%s\n' "$started_at" "$$" >"$STATUS"

set +e
ulimit -n 65535
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=120s 7200 \
  "$PYTHON" "$PIPELINE" \
  --table "$TABLE" \
  --output-table "$TABLE" \
  --target-root "$TARGET" \
  --timepoints t0,seasonal,year \
  --center-mode point \
  --stac-dn-add-override 1000 \
  --legacy-r20m-layout \
  --workers 24 \
  --legacy-config "$LEGACY_CONFIG" \
  --product-scratch-dir "$PRODUCT_CACHE" \
  --crop-scratch-dir "$CROP_CACHE" \
  --existing-product-roots "$EXISTING_PRODUCT_ROOTS" \
  --cdse-env-index 0 \
  --auth-retries 5 \
  --sync-interval 50 \
  --existing-cache-mode adaptive \
  --full-stage-min-tasks 24 \
  --missing-source hybrid \
  --aws-band-workers 4 \
  --aws-require-exact \
  --aws-read-retries 4 \
  --node-band-workers 2 \
  --node-global-workers 12 \
  --node-request-timeout 300 \
  2>&1 | tee "$LOG"
exit_code=${PIPESTATUS[0]}
set -e

finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
finished_epoch=$(date +%s)
elapsed_seconds=$((finished_epoch - started_epoch))
printf 'finished exit_code=%s started_at=%s finished_at=%s elapsed_seconds=%s\n' \
  "$exit_code" "$started_at" "$finished_at" "$elapsed_seconds" >"$STATUS"
exit "$exit_code"
