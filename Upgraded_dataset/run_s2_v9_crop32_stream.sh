#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon/Upgraded_dataset/s2_historical_bounds_native_v9
TRAIN_CSV="$ROOT/temporal_split/s2_v9_train.csv"
TEST_CSV="$ROOT/temporal_split/s2_v9_test.csv"
LOCAL_ROOT=/diniuvol/yuyao/s2_historical_bounds_native_v9_crop32_stage
SOURCE_CACHE=/diniuvol/yuyao/s2_historical_bounds_native_v9_crop_source_cache
REMOTE_ROOT=/transferdiniu2/yuyao/final_crop/s2_historical_bounds_native_32_v9
TRAINER_CACHE=/diniuvol/yuyao/s2_v9_vit_cache
STATE="$ROOT/upload32_state.json"
CROP_LOG="$ROOT/crop32.log"
UPLOAD_LOG="$ROOT/upload32.log"
STATUS="$ROOT/crop32.status"
PIPELINE=/home/yuyao/panopticon/Upgraded_dataset/s2_6time_cdse_legacy512_rebuild.py
UPLOADER=/home/yuyao/panopticon/Upgraded_dataset/stream_s2_crop32_to_remote.py
PYTHON=/home/yuyao/miniconda3/envs/methane/bin/python

mkdir -p "$ROOT" "$LOCAL_ROOT" "$SOURCE_CACHE" "$REMOTE_ROOT" "$TRAINER_CACHE"
started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
started_epoch=$(date +%s)
printf 'running started_at=%s pid=%s\n' "$started_at" "$$" >"$STATUS"

uploader_pid=
cleanup() {
  if [[ -n "${uploader_pid}" ]] && kill -0 "${uploader_pid}" 2>/dev/null; then
    kill "${uploader_pid}" 2>/dev/null || true
    wait "${uploader_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PYTHONUNBUFFERED=1 "$PYTHON" "$UPLOADER" \
  --local-root "$LOCAL_ROOT" \
  --remote-root "$REMOTE_ROOT" \
  --state-json "$STATE" \
  --workers 64 \
  --batch-plumes 16 \
  --buffer-mb 8 \
  --poll-seconds 5 \
  --trainer-cache-dir "$TRAINER_CACHE" \
  --delete-local \
  --no-initialize-from-remote \
  >"$UPLOAD_LOG" 2>&1 &
uploader_pid=$!

set +e
PYTHONUNBUFFERED=1 "$PYTHON" "$PIPELINE" crop-32 \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --out-32-root "$LOCAL_ROOT" \
  --workers 32 \
  --local-cache-dir "$SOURCE_CACHE" \
  --cache-copy-buffer-mb 8 \
  --no-cache-stage-outputs \
  --progress-every 25 \
  --resume \
  2>&1 | tee "$CROP_LOG"
crop_exit=${PIPESTATUS[0]}
set -e

upload_exit=1
if [[ "$crop_exit" -eq 0 ]]; then
  set +e
  wait "$uploader_pid"
  upload_exit=$?
  set -e
  uploader_pid=
fi

finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
finished_epoch=$(date +%s)
elapsed_seconds=$((finished_epoch - started_epoch))
printf 'finished crop_exit=%s upload_exit=%s started_at=%s finished_at=%s elapsed_seconds=%s\n' \
  "$crop_exit" "$upload_exit" "$started_at" "$finished_at" "$elapsed_seconds" \
  >"$STATUS"
if [[ "$crop_exit" -ne 0 ]]; then
  exit "$crop_exit"
fi
exit "$upload_exit"
