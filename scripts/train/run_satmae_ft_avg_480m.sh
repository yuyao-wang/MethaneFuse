#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV to your training manifest CSV}
TEST_CSV=${TEST_CSV:?Set TEST_CSV to your validation/test manifest CSV}
SATMAE_REPO=${SATMAE_REPO:?Set SATMAE_REPO to a local SatMAE checkout}
SATMAE_PRETRAINED=${SATMAE_PRETRAINED:?Set SATMAE_PRETRAINED to the SatMAE checkpoint}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/satmae_ft_avg_480m}
RUN_NAME=${RUN_NAME:-satmae_ft_avg_480m}

"$PYTHON_BIN" baselines/satmae_ft_avg_fusion_480m.py \
  --run_name "$RUN_NAME" \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --satmae_repo "$SATMAE_REPO" \
  --pretrained "$SATMAE_PRETRAINED" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --batch_size "${BATCH_SIZE:-6}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-10}" \
  --accum_steps "${ACCUM_STEPS:-2}" \
  --epochs "${EPOCHS:-7}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --device "${DEVICE:-cuda}" \
  "$@"
