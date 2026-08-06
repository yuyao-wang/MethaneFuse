#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
source_dir="/mnt/engg-niulab/Yuyao/preprocessed_512/L89/l89_6time_temporal_16_resized_to_224"
log_dir="$repo_root/logs/l89_ablation_20260722"
run_name="l89_b1_b5"

mkdir -p "$log_dir" "$repo_root/checkpoints/l89_ablation_20260722"

echo "[$(date -u +%FT%TZ)] starting $run_name on cuda:1 with synchronous local cache" \
  | tee -a "$log_dir/${run_name}.log"

"$python_bin" "$repo_root/Upgraded_dataset/dino_classifier_head_l89_temporal_satmae.py" \
  --train_csv "$source_dir/L89_temporal_train.csv" \
  --test_csv "$source_dir/L89_temporal_test.csv" \
  --weights "$repo_root/weights/panopticon_vitb14_teacher.pth" \
  --band_indices 0,1,2,3,4 \
  --batch_size 13 \
  --epochs 50 \
  --head_lr 1e-3 \
  --temporal_lr 1e-3 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --use_precomputed_stats \
  --seed 20251031 \
  --device cuda:1 \
  --log_interval 100 \
  --local_cache_dir /diniuvol/yuyao/l89_temporal_cache \
  --local_cache_mode sync \
  --local_cache_min_free_gb 200 \
  --checkpoint_dir "$repo_root/checkpoints/l89_ablation_20260722" \
  --run_name "$run_name" \
  2>&1 | tee -a "$log_dir/${run_name}.log"

echo "[$(date -u +%FT%TZ)] finished $run_name" | tee -a "$log_dir/${run_name}.log"
