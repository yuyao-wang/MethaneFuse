#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/yuyao/panopticon
RECROP_SESSION=s2_exact_recrop
DOWNSTREAM_SESSION=s2_exact_downstream
VIT_SESSION=s2_exact_vit_gate
RECROP_LOG="$ROOT/Upgraded_dataset/s2_exact_point_center_v3_recrop.log"
DOWNSTREAM_LOG="$ROOT/Upgraded_dataset/s2_exact_downstream.log"
VIT_MASTER_LOG="$ROOT/Upgraded_dataset/s2_exact_vit_gate_master.log"
SUPERVISOR_LOG="$ROOT/Upgraded_dataset/s2_exact_supervisor.log"
SOURCE_CSV=/home/yuyao/methane_train/Upgrade_data_pipeline/csv/s2_6time_point_covering_tiles_v3.csv
RECROP_CSV=/home/yuyao/methane_train/Upgrade_data_pipeline/csv/s2_6time_point_center_exact_v3_paths.csv
METHANE_PY=/home/yuyao/miniconda3/envs/methane/bin/python
LEGACY_FILL="$ROOT/Upgraded_dataset/s2_fill_legacy_timepoints_v2.py"
TILE_RESOLVER="$ROOT/Upgraded_dataset/s2_resolve_point_covering_tiles.py"
TILE_REPORT="$ROOT/Upgraded_dataset/s2_exact_reports_v3/tile_resolution.json"
TILE_LOG="$ROOT/Upgraded_dataset/s2_exact_tile_resolution_v3.log"
EXISTING_PRODUCT_ROOTS="/diniuvol/yuyao/s2_cdse_point_repair_cache,/diniuvol/yuyao/s2_early_boundary_products,/mnt/engg-niulab/yuyao/sensors_raw_data/S2/raw_data_dir_s2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2_90360,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7"
MAX_RECROP_SECONDS=7200
POLL_SECONDS=60

stamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "$SUPERVISOR_LOG"
}

session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}

recrop_counts() {
  "$METHANE_PY" - "$RECROP_CSV" <<'PY'
import json
import sys

import pandas as pd

frame = pd.read_csv(sys.argv[1], low_memory=False)
timepoints = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
allowed = {"downloaded", "target_exists"}
complete = 0
failed = 0
pending = 0
for timepoint in timepoints:
    values = frame[f"{timepoint}_recrop_status"].fillna("").astype(str)
    complete += int(values.isin(allowed).sum())
    pending += int((values == "").sum())
    failed += int((~values.isin(allowed | {""})).sum())
print(json.dumps({
    "complete": complete,
    "failed": failed,
    "pending": pending,
    "total": len(frame) * len(timepoints),
}))
PY
}

start_recrop() {
  local node_workers=$1
  printf '\n===== SUPERVISOR_RESTART_NODE%s %s =====\n' \
    "$node_workers" "$(stamp)" >> "$RECROP_LOG"
  tmux new-session -d -s "$RECROP_SESSION" \
    "cd /home/yuyao/methane_train && ulimit -n 65535 && exec env PYTHONUNBUFFERED=1 $METHANE_PY $ROOT/Upgraded_dataset/s2_exact_point_recrop.py --table $SOURCE_CSV --output-table $RECROP_CSV --target-root /mnt/engg-niulab/yuyao/sensors_raw_data/S2_point_center_exact_v3 --timepoints t0,prev1,prev2,prev3,seasonal,year --workers 32 --legacy-config /home/yuyao/methane_train/data_preprocess/configs/carbon_mapper_sentinel2_plume_download.yaml --product-scratch-dir /diniuvol/yuyao/s2_point_center_recrop_cache/products --crop-scratch-dir /diniuvol/yuyao/s2_point_center_recrop_cache/crops_v3 --existing-product-roots $EXISTING_PRODUCT_ROOTS --cdse-env-index 0 --auth-retries 5 --sync-interval 100 --existing-cache-mode adaptive --full-stage-min-tasks 32 --missing-source hybrid --aws-band-workers 4 --aws-require-exact --aws-read-retries 4 --node-band-workers 4 --node-global-workers $node_workers --node-request-timeout 300 >> $RECROP_LOG 2>&1"
}

wait_for_recrop() {
  local started=$SECONDS
  local last_report=$SECONDS
  local retries=0
  local node_workers=32

  while true; do
    if ! session_exists "$RECROP_SESSION" && [[ ! -f "$RECROP_CSV" ]]; then
      log "RECROP_START node_global_workers=$node_workers"
      start_recrop "$node_workers"
      sleep 240
      continue
    fi

    if session_exists "$RECROP_SESSION"; then
      if (( SECONDS - started > MAX_RECROP_SECONDS )); then
        log "FAIL recrop exceeded ${MAX_RECROP_SECONDS}s"
        tmux kill-session -t "$RECROP_SESSION" 2>/dev/null || true
        return 1
      fi
      if (( SECONDS - last_report >= 300 )); then
        log "RECROP $(recrop_counts)"
        last_report=$SECONDS
      fi
      sleep "$POLL_SECONDS"
      continue
    fi

    log "LEGACY_FILL_START"
    "$METHANE_PY" "$LEGACY_FILL" \
      --csv "$RECROP_CSV" \
      --legacy-root /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_raw_s2_90360_fixed_512 \
      --target-root /mnt/engg-niulab/yuyao/sensors_raw_data/S2_point_center_exact_v3 \
      --scratch-root /diniuvol/yuyao/s2_point_center_recrop_cache/legacy_v3 \
      --workers 16
    log "LEGACY_FILL_DONE"

    local counts
    counts=$(recrop_counts)
    log "RECROP_EXIT $counts"
    if "$METHANE_PY" - "$counts" <<'PY'
import json
import sys

counts = json.loads(sys.argv[1])
raise SystemExit(0 if counts["complete"] == counts["total"] and counts["failed"] == 0 else 1)
PY
    then
      return 0
    fi
    if "$METHANE_PY" - "$counts" <<'PY'
import json
import sys

counts = json.loads(sys.argv[1])
raise SystemExit(0 if counts["failed"] > 0 else 1)
PY
    then
      log "FAIL recrop has failed tasks; repair manifest before retry"
      return 1
    fi

    retries=$((retries + 1))
    if (( retries > 3 )); then
      log "FAIL recrop incomplete after ${retries} exits"
      return 1
    fi
    if (( retries == 2 )); then
      node_workers=24
    elif (( retries == 3 )); then
      node_workers=16
    fi
    log "RECROP_RETRY attempt=$retries node_global_workers=$node_workers"
    start_recrop "$node_workers"
    sleep 240
  done
}

wait_for_session() {
  local session=$1
  local label=$2
  local maximum=$3
  local started=$SECONDS
  local last_report=$SECONDS

  while session_exists "$session"; do
    if (( SECONDS - started > maximum )); then
      log "FAIL $label exceeded ${maximum}s"
      tmux kill-session -t "$session" 2>/dev/null || true
      return 1
    fi
    if (( SECONDS - last_report >= 300 )); then
      log "RUNNING $label elapsed=$((SECONDS - started))s"
      last_report=$SECONDS
    fi
    sleep "$POLL_SECONDS"
  done
}

: > "$SUPERVISOR_LOG"
log "SUPERVISOR_START"

if [[ ! -f "$SOURCE_CSV" ]]; then
  mkdir -p "$(dirname "$TILE_REPORT")"
  : > "$TILE_LOG"
  log "TILE_RESOLUTION_START"
  timeout --signal=TERM --kill-after=60s 7200 \
    "$METHANE_PY" "$TILE_RESOLVER" \
    --input-csv /home/yuyao/methane_train/Upgrade_data_pipeline/csv/s2_6time_all6_available_paths.csv \
    --output-csv "$SOURCE_CSV" \
    --report "$TILE_REPORT" \
    --existing-roots "$EXISTING_PRODUCT_ROOTS" \
    --workers 32 \
    --progress-every 100 \
    2>&1 | tee "$TILE_LOG"
  log "TILE_RESOLUTION_DONE"
fi

if ! wait_for_recrop; then
  exit 1
fi
log "RECROP_COMPLETE"

rm -f "$DOWNSTREAM_LOG"
tmux new-session -d -s "$DOWNSTREAM_SESSION" \
  "exec $ROOT/Upgraded_dataset/run_s2_exact_downstream.sh >> $DOWNSTREAM_LOG 2>&1"
log "DOWNSTREAM_STARTED"
if ! wait_for_session "$DOWNSTREAM_SESSION" downstream 18000; then
  exit 1
fi
if ! rg -q 'PIPELINE_READY_FOR_VIT' "$DOWNSTREAM_LOG"; then
  log "FAIL downstream exited before quality gate passed"
  exit 1
fi
log "DOWNSTREAM_COMPLETE"

rm -f "$VIT_MASTER_LOG"
tmux new-session -d -s "$VIT_SESSION" \
  "exec $ROOT/Upgraded_dataset/run_s2_exact_vit_gate.sh >> $VIT_MASTER_LOG 2>&1"
log "VIT_GATE_STARTED"
if ! wait_for_session "$VIT_SESSION" vit_gate 15000; then
  exit 1
fi
if ! rg -q 'VIT_GATE_PASSED' "$VIT_MASTER_LOG"; then
  log "FAIL ViT gate did not pass"
  exit 1
fi

log "VIT_GATE_COMPLETE"
log "SUPERVISOR_COMPLETE"
