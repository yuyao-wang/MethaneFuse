#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/yuyao/miniconda3/envs/panopticon/bin/python
REPO=/home/yuyao/panopticon
DATA=${DATA:-/diniuvol/yuyao/s2_6time_legacy_notebook_point_v14_32}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/transferdiniu2/yuyao/checkpoints/s2_point_center_v14}
RUN_NAME=${RUN_NAME:-s2_6time_point_center_v14_concat_full_finetune}
EPOCHS=${EPOCHS:-3}

cd "$REPO"
exec "$PYTHON" \
  Upgraded_dataset/dino_classifier_head_s2_temporal_satmae.py \
  --train_csv "$DATA/train.csv" \
  --test_csv "$DATA/test.csv" \
  --weights /home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth \
  --train_backbone \
  --temporal_fusion concat_channels \
  --path_columns path_t0,path_prev1,path_prev2,path_prev3,path_seasonal,path_year \
  --time_columns t0_image_time,prev1_image_time,prev2_image_time,prev3_image_time,seasonal_image_time,year_image_time \
  --channel_indices 0,1,2,3,4,5,6,7,10,11 \
  --input_resize_size 224 \
  --batch_size 32 \
  --epochs "$EPOCHS" \
  --num_workers 16 \
  --prefetch_factor 4 \
  --stats_samples 2000 \
  --stats_workers 16 \
  --local_cache_mode off \
  --checkpoint_dir "$CHECKPOINT_ROOT" \
  --run_name "$RUN_NAME" \
  --log_interval 100 \
  --device cuda:0 \
  "$@"
