#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV to your training manifest CSV}
TEST_CSV=${TEST_CSV:?Set TEST_CSV to your validation/test manifest CSV}
WEIGHTS=${WEIGHTS:-weights/panopticon_vitb14_teacher.pth}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/methanefuse_pretrain}
RUN_NAME=${RUN_NAME:-methanefuse_pretrain}
DEVICE=${DEVICE:-cuda}
CACHE_DIR=${CACHE_DIR:-}
USE_WANDB=${USE_WANDB:-0}

WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_ARGS+=(--use_wandb --wandb_project "${WANDB_PROJECT:-methanefuse}" --wandb_run_name "$RUN_NAME")
fi

CACHE_ARGS=()
if [[ -n "$CACHE_DIR" ]]; then
  CACHE_ARGS+=(--local_cache_dir "$CACHE_DIR" --local_cache_min_free_gb "${LOCAL_CACHE_MIN_FREE_GB:-20}")
  if [[ "${LOCAL_CACHE_WARMUP:-0}" == "1" ]]; then
    CACHE_ARGS+=(--local_cache_warmup --local_cache_workers "${LOCAL_CACHE_WORKERS:-8}")
  fi
fi

"$PYTHON_BIN" src/models/pretrain_multisensor.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --weights "$WEIGHTS" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --batch_size "${BATCH_SIZE:-12}" \
  --epochs "${EPOCHS:-30}" \
  --train_backbone \
  --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS:-1}" \
  --backbone_lr "${BACKBONE_LR:-5e-5}" \
  --head_lr "${HEAD_LR:-1e-3}" \
  --weight_decay "${WEIGHT_DECAY:-5e-4}" \
  --sensor_aux_loss_weight "${SENSOR_AUX_LOSS_WEIGHT:-0.3}" \
  --row_fusion_mode "${ROW_FUSION_MODE:-max}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --device "$DEVICE" \
  "${WANDB_ARGS[@]}" \
  "${CACHE_ARGS[@]}" \
  "$@"
