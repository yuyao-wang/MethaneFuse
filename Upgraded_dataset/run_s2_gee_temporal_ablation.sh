#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon
PYTHON=/home/yuyao/miniconda3/envs/panopticon/bin/python
DATA_ROOT="$ROOT/Upgraded_dataset/s2_gee_legacy_notebook_6time/temporal_cutoff_split/cutoff_2025-12-22"
CHECKPOINT_ROOT=/diniuvol/yuyao/checkpoints/s2_gee_legacy_notebook_6time
OUTPUT_ROOT="$ROOT/Upgraded_dataset/s2_gee_legacy_notebook_6time/temporal_ablation"

mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_ROOT"
cd "$ROOT"

run_ablation() {
  local run_name=$1
  local path_columns=$2
  local time_columns=$3
  local log="$OUTPUT_ROOT/${run_name}.log"
  local status="$OUTPUT_ROOT/${run_name}.status"
  local started_at
  local started_epoch

  started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  started_epoch=$(date +%s)
  printf 'running started_at=%s pid=%s\n' "$started_at" "$$" >"$status"

  set +e
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PYTHON" \
    Upgraded_dataset/dino_classifier_head_s2_temporal_satmae.py \
    --train_csv "$DATA_ROOT/train.csv" \
    --test_csv "$DATA_ROOT/test.csv" \
    --weights "$ROOT/weights/panopticon_vitb14_teacher.pth" \
    --train_backbone \
    --temporal_fusion concat_channels \
    --path_columns "$path_columns" \
    --time_columns "$time_columns" \
    --channel_indices all \
    --input_resize_size 224 \
    --input_resize_align_corners \
    --batch_size 32 \
    --epochs 3 \
    --head_lr 0.001 \
    --backbone_lr 0.0001 \
    --lr_scheduler noam \
    --warmup_steps 4000 \
    --weight_decay 0.0005 \
    --momentum 0.9 \
    --num_workers 4 \
    --prefetch_factor 1 \
    --stats_samples 2000 \
    --stats_workers 4 \
    --stats_seed 73 \
    --local_cache_mode off \
    --local_cache_dir "" \
    --checkpoint_dir "$CHECKPOINT_ROOT" \
    --run_name "$run_name" \
    --device cuda:0 \
    --num_gpus 1 \
    --log_interval 100 \
    2>&1 | tee -a "$log"
  local exit_code=${PIPESTATUS[0]}
  set -e

  local finished_at
  local elapsed_seconds
  finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  elapsed_seconds=$(($(date +%s) - started_epoch))
  printf 'finished exit_code=%s started_at=%s finished_at=%s elapsed_seconds=%s\n' \
    "$exit_code" "$started_at" "$finished_at" "$elapsed_seconds" >"$status"
  return "$exit_code"
}

run_ablation \
  s2_gee_t0_only_all12_concat_gate3 \
  path_t0 \
  t0_image_time

run_ablation \
  s2_gee_legacy3_all12_concat_gate3 \
  path_t0,path_seasonal,path_year \
  t0_image_time,seasonal_image_time,year_image_time
