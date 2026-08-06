#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon/Upgraded_dataset/s2_historical_bounds_native_v9
LOG="$ROOT/finalize.log"
STATUS="$ROOT/finalize.status"

started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
started_epoch=$(date +%s)
printf 'running started_at=%s pid=%s\n' "$started_at" "$$" >"$STATUS"

set +e
PYTHONUNBUFFERED=1 /home/yuyao/miniconda3/envs/methane/bin/python \
  /home/yuyao/panopticon/Upgraded_dataset/s2_finalize_bounds_native_v9.py \
  --workers 32 \
  --content-samples 3000 \
  --max-drop-rows 400 \
  2>&1 | tee "$LOG"
exit_code=${PIPESTATUS[0]}
set -e

finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
finished_epoch=$(date +%s)
elapsed_seconds=$((finished_epoch - started_epoch))
printf 'finished exit_code=%s started_at=%s finished_at=%s elapsed_seconds=%s\n' \
  "$exit_code" "$started_at" "$finished_at" "$elapsed_seconds" >"$STATUS"
exit "$exit_code"
