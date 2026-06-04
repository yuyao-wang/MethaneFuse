#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV to your training manifest CSV}
TEST_CSV=${TEST_CSV:?Set TEST_CSV to your validation/test manifest CSV}
WEIGHTS=${WEIGHTS:?Set WEIGHTS to the Panopticon/DINOv2 backbone checkpoint}
OUT_ROOT=${OUT_ROOT:-checkpoints/per_vit}
SENSORS=${SENSORS:-s2,l89,s5p,wv3}

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

  "$PYTHON_BIN" "$script" \
    --train_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV" \
    --weights "$WEIGHTS" \
    --checkpoint_dir "$OUT_ROOT/$sensor" \
    "$@"
}

IFS=',' read -ra SENSOR_LIST <<< "$SENSORS"
for sensor in "${SENSOR_LIST[@]}"; do
  run_sensor "$sensor" "$@"
done
