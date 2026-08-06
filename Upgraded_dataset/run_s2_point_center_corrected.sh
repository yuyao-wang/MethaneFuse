#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

cd /home/yuyao/panopticon

/home/yuyao/miniconda3/envs/panopticon/bin/python -u \
  Upgraded_dataset/dino_classifier_head_s2_temporal_satmae.py \
  --train_csv /diniuvol/yuyao/s2_6time_point_center_corrected_32/train.csv \
  --test_csv /diniuvol/yuyao/s2_6time_point_center_corrected_32/test.csv \
  --weights /home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth \
  --train_backbone \
  --channel_indices all \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --epochs 50 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --stats_samples 2000 \
  --stats_workers 16 \
  --input_resize_size 224 \
  --local_cache_mode off \
  --checkpoint_dir /transferdiniu2/yuyao/checkpoints/s2_point_center_corrected \
  --run_name s2_6time_point_center_corrected_full_finetune \
  --device cuda:0 \
  --log_interval 100 \
  "$@" 2>&1 | tee -a \
  /home/yuyao/panopticon/Upgraded_dataset/s2_point_center_corrected_train.log
