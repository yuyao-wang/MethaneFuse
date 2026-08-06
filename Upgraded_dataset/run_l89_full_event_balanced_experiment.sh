#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <gpu> <run_name> <train_csv>" >&2
  exit 2
fi

gpu="$1"
run_name="$2"
train_csv="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
split_dir="$repo_root/Upgraded_dataset/l89_6time_full_event_balanced_split"
log_dir="$repo_root/logs/l89_full_event_balanced"
mkdir -p "$log_dir"

cd "$repo_root"
exec /home/yuyao/miniconda3/envs/panopticon/bin/python \
  Upgraded_dataset/dino_classifier_head_l89_temporal_satmae.py \
  --train_csv "$train_csv" \
  --test_csv "$split_dir/L89_temporal_test_full_event_balanced.csv" \
  --band_indices 0,1,2,3,4,5,6 \
  --batch_size 24 \
  --epochs 8 \
  --head_lr 1e-3 \
  --temporal_lr 1e-3 \
  --lr_scheduler noam \
  --warmup_steps 1500 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --device "cuda:$gpu" \
  --seed 20251031 \
  --use_precomputed_stats \
  --local_cache_dir /diniuvol/yuyao/l89_temporal_cache \
  --local_cache_mode sync \
  --cache_prefetch_rows 0 \
  --local_cache_min_free_gb 200 \
  --checkpoint_dir checkpoints/l89_full_event_balanced \
  --run_name "$run_name" \
  --log_interval 100 \
  2>&1
