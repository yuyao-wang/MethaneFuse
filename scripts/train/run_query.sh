#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV to your training manifest CSV}
TEST_CSV=${TEST_CSV:?Set TEST_CSV to your validation/test manifest CSV}
WEIGHTS=${WEIGHTS:?Set WEIGHTS to the Panopticon/DINOv2 backbone checkpoint}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/methanefuse_pretrain}
RUN_NAME=${RUN_NAME:-methanefuse_pretrain}

"$PYTHON_BIN" src/models/pretrain_multisensor.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --weights "$WEIGHTS" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb_run_name "$RUN_NAME" \
  "$@"
