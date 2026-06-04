#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV to your training manifest CSV}
TEST_CSV=${TEST_CSV:?Set TEST_CSV to your validation/test manifest CSV}
ANYSAT_REPO=${ANYSAT_REPO:?Set ANYSAT_REPO to a local AnySat checkout}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/anysat_ft_avg_480m}
RUN_NAME=${RUN_NAME:-anysat_ft_avg_480m}
DEVICE=${DEVICE:-cuda}
PROCESSED_CACHE_DIR=${PROCESSED_CACHE_DIR:-}
USE_WANDB=${USE_WANDB:-0}

WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_ARGS+=(--use_wandb --wandb_project "${WANDB_PROJECT:-methanefuse_baselines}" --wandb_run_name "$RUN_NAME")
fi

"$PYTHON_BIN" baselines/anysat_ft_avg_fusion_480m.py \
  --run_name "$RUN_NAME" \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --anysat_repo "$ANYSAT_REPO" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --batch_size "${BATCH_SIZE:-8}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-12}" \
  --epochs "${EPOCHS:-7}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --device "$DEVICE" \
  --lr_anysat "${LR_ANYSAT:-1e-5}" \
  --lr_head "${LR_HEAD:-1e-3}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.05}" \
  --patch_size "${PATCH_SIZE:-20}" \
  --freeze_anysat_epochs "${FREEZE_ANYSAT_EPOCHS:-1}" \
  --processed_cache_dir "$PROCESSED_CACHE_DIR" \
  --processed_cache_min_free_gb "${PROCESSED_CACHE_MIN_FREE_GB:-20}" \
  "${WANDB_ARGS[@]}" \
  "$@"
