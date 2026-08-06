#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/yuyao/panopticon"
SESSION="s2_cache_train"
PYTHON="/home/yuyao/miniconda3/envs/panopticon/bin/python"
CACHE_SCRIPT="${ROOT}/Upgraded_dataset/cache_s2_crop32_local.py"
TRAIN_SCRIPT="${ROOT}/Upgraded_dataset/train_s2_6time_cdse_legacy512_exact.sh"
CACHE_ROOT="/diniuvol/yuyao/s2_6time_cdse_legacy512_exact_cache/input32"
CACHE_LOG="${ROOT}/Upgraded_dataset/s2_cdse_legacy512_cache32.log"
TRAIN_LOG="${ROOT}/Upgraded_dataset/s2_cdse_legacy512_train.log"
WATCH_LOG="${ROOT}/Upgraded_dataset/s2_cdse_legacy512_watchdog.log"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${WATCH_LOG}"
}

cache_complete() {
  [[ -s "${CACHE_ROOT}/train_patches_32_local.csv" ]] \
    && [[ -s "${CACHE_ROOT}/test_patches_32_local.csv" ]]
}

training_complete() {
  [[ -s "${TRAIN_LOG}" ]] && grep -q '^Epoch 50:' "${TRAIN_LOG}"
}

start_cache_and_train() {
  tmux new-session -d -s "${SESSION}" \
    "cd '${ROOT}' && '${PYTHON}' '${CACHE_SCRIPT}' --workers 224 >> '${CACHE_LOG}' 2>&1 && bash '${TRAIN_SCRIPT}' >> '${TRAIN_LOG}' 2>&1"
  log "restarted cache pipeline in tmux ${SESSION}"
}

start_training_resume() {
  tmux new-session -d -s "${SESSION}" \
    "cd '${ROOT}' && bash '${TRAIN_SCRIPT}' --resume >> '${TRAIN_LOG}' 2>&1"
  log "restarted training with --resume in tmux ${SESSION}"
}

log "watchdog started"
while true; do
  if training_complete; then
    log "training reached epoch 50; watchdog exiting"
    exit 0
  fi

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    sleep 30
    continue
  fi

  log "tmux ${SESSION} is absent"
  sleep 5
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    continue
  fi

  if cache_complete; then
    start_training_resume
  else
    start_cache_and_train
  fi
  sleep 60
done
