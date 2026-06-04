#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV to your training manifest CSV}
TEST_CSV=${TEST_CSV:?Set TEST_CSV to your validation/test manifest CSV}
WEIGHTS=${WEIGHTS:-weights/panopticon_vitb14_teacher.pth}
OUT_ROOT=${OUT_ROOT:-checkpoints/per_vit}
SENSORS=${SENSORS:-s2,l89,s5p,wv3}
DEVICE=${DEVICE:-cuda}
CACHE_DIR=${CACHE_DIR:-}
USE_WANDB=${USE_WANDB:-0}

run_sensor() {
  local sensor="$1"
  local script
  case "$sensor" in
    s2) script="baselines/per_vit/s2_temporal.py" ;;
    l89) script="baselines/per_vit/l89_temporal.py" ;;
    s5p) script="baselines/per_vit/s5p_temporal.py" ;;
    wv3|emit) script="baselines/per_vit/emit_wv3_temporal.py" ;;
    *) echo "Unknown sensor: $sensor" >&2; return 1 ;;
  esac

  local args=(
    --train_csv "$TRAIN_CSV"
    --test_csv "$TEST_CSV"
    --weights "$WEIGHTS"
    --checkpoint_dir "$OUT_ROOT/$sensor"
    --batch_size "${BATCH_SIZE:-64}"
    --epochs "${EPOCHS:-50}"
    --head_lr "${HEAD_LR:-1e-3}"
    --weight_decay "${WEIGHT_DECAY:-5e-4}"
    --num_workers "${NUM_WORKERS:-8}"
    --device "$DEVICE"
  )

  if [[ "$USE_WANDB" == "1" ]]; then
    args+=(--use_wandb --wandb_project "${WANDB_PROJECT:-methanefuse_baselines}" --wandb_run_name "per_vit_${sensor}")
  fi

  if [[ -n "$CACHE_DIR" ]]; then
    args+=(--local_cache_dir "$CACHE_DIR")
    if [[ "${LOCAL_CACHE_WARMUP:-0}" == "1" ]]; then
      args+=(--local_cache_warmup --local_cache_workers "${LOCAL_CACHE_WORKERS:-8}")
    fi
  fi

  "$PYTHON_BIN" "$script" "${args[@]}" "$@"
}

IFS=',' read -ra SENSOR_LIST <<< "$SENSORS"
for sensor in "${SENSOR_LIST[@]}"; do
  run_sensor "$sensor" "$@"
done
