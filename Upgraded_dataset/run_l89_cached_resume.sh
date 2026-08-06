#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <gpu> <run_name> <train_csv> <test_csv> <band_indices>" >&2
  exit 2
fi

gpu="$1"
run_name="$2"
train_csv="$3"
test_csv="$4"
band_indices="$5"
repo_root="/home/yuyao/panopticon"
log_dir="$repo_root/logs/l89_ablation_20260722"

echo "[$(date -u +%FT%TZ)] resuming $run_name on cuda:$gpu with synchronous shared cache" \
  | tee -a "$log_dir/${run_name}.log"

"/home/yuyao/miniconda3/envs/panopticon/bin/python" \
  "$repo_root/Upgraded_dataset/dino_classifier_head_l89_temporal_satmae.py" \
  --train_csv "$train_csv" \
  --test_csv "$test_csv" \
  --weights "$repo_root/weights/panopticon_vitb14_teacher.pth" \
  --band_indices "$band_indices" \
  --batch_size 13 \
  --epochs 5 \
  --head_lr 1e-3 \
  --temporal_lr 1e-3 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --use_precomputed_stats \
  --seed 20251031 \
  --device "cuda:$gpu" \
  --log_interval 100 \
  --local_cache_dir /diniuvol/yuyao/l89_temporal_cache \
  --local_cache_mode sync \
  --local_cache_min_free_gb 200 \
  --checkpoint_dir "$repo_root/checkpoints/l89_ablation_20260722" \
  --run_name "$run_name" \
  --resume \
  2>&1 | tee -a "$log_dir/${run_name}.log"

