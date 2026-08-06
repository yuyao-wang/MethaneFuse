#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/yuyao/panopticon"
METRICS="${REPO_ROOT}/Upgraded_dataset/s2_legacy360_matched_gee_splits/reproduction_results/event_disjoint_80_20.cdse_compat.metrics.jsonl"

while true; do
  lines=0
  if [[ -f "${METRICS}" ]]; then
    lines="$(wc -l < "${METRICS}")"
  fi
  if (( lines >= 2 )); then
    break
  fi
  echo "[Queue] Waiting for CDSE-compat event metrics (${lines}/2 epochs)."
  sleep 60
done

cd "${REPO_ROOT}"
GPU_INDEX="${GPU_INDEX:-1}" \
EPOCHS="${EPOCHS:-2}" \
BATCH_SIZE="${BATCH_SIZE:-32}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
bash Upgraded_dataset/run_s2_legacy360_t0_only_reproduction.sh
