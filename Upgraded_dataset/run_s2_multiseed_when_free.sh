#!/usr/bin/env bash
#
# Launch a multiseed replicate on GPU0 only once the card has enough free
# memory for a full batch-32 finetune (~30 GiB observed). GPU0 is shared, so
# starting unconditionally either OOMs this run or someone else's.
#
# Usage: run_s2_multiseed_when_free.sh <legacy|gee> <seed> [required_MiB]

set -euo pipefail
side="${1:?usage: $0 <legacy|gee> <seed> [required_MiB]}"
seed="${2:?usage: $0 <legacy|gee> <seed> [required_MiB]}"
required="${3:-32000}"
root="/home/yuyao/panopticon/Upgraded_dataset"
log="${root}/s2_legacy360_matched_gee_splits/multiseed_20260731/${side}_seed${seed}.driver.log"

while true; do
  read -r total used < <(nvidia-smi --id=0 --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr ',' ' ')
  free=$(( total - used ))
  if (( free >= required )); then
    # Serialize check-and-launch. Two chains polling independently once read the
    # same free memory and started together, and the loser OOMed. The lock is
    # held for a few minutes past launch so the winner's allocation is already
    # visible when the next chain re-reads memory.used.
    exec 9>/tmp/s2_multiseed_gpu0.lock
    if flock -n 9; then
      read -r total used < <(nvidia-smi --id=0 --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr ',' ' ')
      if (( total - used >= required )); then
        echo "[gate $(date -u +%FT%TZ)] $(( total - used )) MiB free >= ${required}; launching ${side} seed ${seed}"
        "${root}/run_s2_rendering_multiseed.sh" "${side}" "${seed}" >>"${log}" 2>&1 &
        train_pid=$!
        sleep 300
        flock -u 9
        wait "${train_pid}"
        exit $?
      fi
      flock -u 9
    fi
    exec 9>&-
  fi
  sleep 120
done
