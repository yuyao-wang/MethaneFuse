#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon/Upgraded_dataset/s2_historical_bounds_native_v9
DATA_ROOT=/transferdiniu2/yuyao/final_crop/s2_historical_bounds_native_32_v9
TRAIN_CSV="$DATA_ROOT/train_patches_32.csv"
TEST_CSV="$DATA_ROOT/test_patches_32.csv"
CACHE=/diniuvol/yuyao/s2_v9_vit_cache
CHECKPOINT_ROOT=/transferdiniu2/yuyao/checkpoints/s2_historical_bounds_native_v9
RUN_NAME=s2_v9_bounds_native_6time_full_finetune_gate
LOG="$ROOT/vit_gate.log"
STATUS="$ROOT/vit_gate.status"
PYTHON=/home/yuyao/miniconda3/envs/panopticon/bin/python
TRAINER=/home/yuyao/panopticon/Upgraded_dataset/dino_classifier_head_s2_temporal_satmae.py

mkdir -p "$ROOT" "$CACHE" "$CHECKPOINT_ROOT"
started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
started_epoch=$(date +%s)
printf 'running started_at=%s pid=%s\n' "$started_at" "$$" >"$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PYTHON" "$TRAINER" \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --weights /home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth \
  --train_backbone \
  --temporal_fusion concat_channels \
  --channel_indices 0,1,2,3,4,5,6,7,10,11 \
  --input_resize_size 224 \
  --batch_size 16 \
  --epochs 3 \
  --head_lr 0.001 \
  --backbone_lr 0.0001 \
  --lr_scheduler noam \
  --warmup_steps 4000 \
  --weight_decay 0.0005 \
  --momentum 0.9 \
  --num_workers 16 \
  --prefetch_factor 2 \
  --stats_samples 2000 \
  --stats_workers 8 \
  --stats_seed 20260724 \
  --local_cache_mode sync \
  --local_cache_dir "$CACHE" \
  --local_cache_warmup \
  --local_cache_workers 64 \
  --checkpoint_dir "$CHECKPOINT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda:0 \
  --num_gpus 1 \
  --log_interval 100 \
  --max_train_steps 2500 \
  --max_eval_steps 1000 \
  2>&1 | tee "$LOG"
exit_code=${PIPESTATUS[0]}
set -e

finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
finished_epoch=$(date +%s)
elapsed_seconds=$((finished_epoch - started_epoch))
printf 'finished exit_code=%s started_at=%s finished_at=%s elapsed_seconds=%s\n' \
  "$exit_code" "$started_at" "$finished_at" "$elapsed_seconds" >"$STATUS"
exit "$exit_code"
