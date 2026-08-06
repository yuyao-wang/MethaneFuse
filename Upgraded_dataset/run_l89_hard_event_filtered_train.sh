#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
split_dir="$repo_root/Upgraded_dataset/l89_6time_temporal_hard_event_filtered_split"

cd "$repo_root"
exec /home/yuyao/miniconda3/envs/panopticon/bin/python \
  Upgraded_dataset/dino_classifier_head_l89_temporal_satmae.py \
  --train_csv "$split_dir/L89_temporal_train_hard_event_filtered.csv" \
  --test_csv "$split_dir/L89_temporal_test_hard_event_filtered.csv" \
  --band_indices 0,1,2,3,4,5,6 \
  --batch_size 24 \
  --epochs 8 \
  --head_lr 1e-3 \
  --temporal_lr 1e-3 \
  --lr_scheduler noam \
  --warmup_steps 4000 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --device cuda:1 \
  --seed 20251031 \
  --use_precomputed_stats \
  --local_cache_dir /diniuvol/yuyao/l89_temporal_cache \
  --local_cache_mode sync \
  --cache_prefetch_rows 0 \
  --local_cache_min_free_gb 200 \
  --checkpoint_dir checkpoints/l89_hard_event_filtered \
  --run_name l89_hard_event_filtered_seed20251031 \
  --log_interval 100 \
  2>&1
