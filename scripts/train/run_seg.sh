#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV}
TEST_CSV=${TEST_CSV:?Set TEST_CSV}
WEIGHTS=${WEIGHTS:-weights/panopticon_vitb14_teacher.pth}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/methanefuse_unet}
RUN_NAME=${RUN_NAME:-methanefuse_unet}

"$PYTHON_BIN" baselines/unet/unet_multisensor_baseline_iou_plus.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --weights "$WEIGHTS" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb_run_name "$RUN_NAME" \
  "$@"
