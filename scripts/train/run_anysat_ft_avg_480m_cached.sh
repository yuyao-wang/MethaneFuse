#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV to your training manifest CSV}
TEST_CSV=${TEST_CSV:?Set TEST_CSV to your validation/test manifest CSV}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/anysat_ft_avg_480m}
RUN_NAME=${RUN_NAME:-anysat_ft_avg_480m}
PROCESSED_CACHE_DIR=${PROCESSED_CACHE_DIR:-}

"$PYTHON_BIN" baselines/anysat_ft_avg_fusion_480m.py \
  --run_name "$RUN_NAME" \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --batch_size "${BATCH_SIZE:-8}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-12}" \
  --epochs "${EPOCHS:-7}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --device "${DEVICE:-cuda}" \
  --processed_cache_dir "$PROCESSED_CACHE_DIR" \
  "$@"
