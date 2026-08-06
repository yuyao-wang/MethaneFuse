#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
train_script="$repo_root/Upgraded_dataset/dino_classifier_head_l89_temporal_satmae.py"
source_dir="/mnt/engg-niulab/Yuyao/preprocessed_512/L89/l89_6time_temporal_16_resized_to_224"
drop_dir="$repo_root/Upgraded_dataset/l89_6time_temporal_drop_3_hard_events"
checkpoint_dir="$repo_root/checkpoints/l89_ablation_20260722"
log_dir="$repo_root/logs/l89_ablation_20260722"

mkdir -p "$checkpoint_dir" "$log_dir"

run_experiment() {
  local gpu="$1"
  local run_name="$2"
  local train_csv="$3"
  local test_csv="$4"
  local band_indices="$5"
  local log_file="$log_dir/${run_name}.log"

  echo "[$(date -u +%FT%TZ)] starting $run_name on cuda:$gpu bands=$band_indices" | tee -a "$log_file"
  "$python_bin" "$train_script" \
    --train_csv "$train_csv" \
    --test_csv "$test_csv" \
    --weights "$repo_root/weights/panopticon_vitb14_teacher.pth" \
    --band_indices "$band_indices" \
    --batch_size 13 \
    --epochs 50 \
    --head_lr 1e-3 \
    --temporal_lr 1e-3 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --use_precomputed_stats \
    --seed 20251031 \
    --device "cuda:$gpu" \
    --log_interval 100 \
    --local_cache_mode off \
    --checkpoint_dir "$checkpoint_dir" \
    --run_name "$run_name" \
    2>&1 | tee -a "$log_file"
  echo "[$(date -u +%FT%TZ)] finished $run_name" | tee -a "$log_file"
}

gpu0_lane_a() {
  run_experiment 0 \
    l89_drop3_full_bands \
    "$drop_dir/L89_temporal_train_drop_3_hard_events.csv" \
    "$drop_dir/L89_temporal_test_drop_3_hard_events.csv" \
    0,1,2,3,4,5,6
  run_experiment 0 \
    l89_b1_b5 \
    "$source_dir/L89_temporal_train.csv" \
    "$source_dir/L89_temporal_test.csv" \
    0,1,2,3,4
}

gpu0_lane_b() {
  run_experiment 0 \
    l89_b6_b7 \
    "$source_dir/L89_temporal_train.csv" \
    "$source_dir/L89_temporal_test.csv" \
    5,6
  run_experiment 0 \
    l89_non_rgb_b1_b5_b6_b7 \
    "$source_dir/L89_temporal_train.csv" \
    "$source_dir/L89_temporal_test.csv" \
    0,4,5,6
}

gpu0_lane_a > "$log_dir/gpu0_lane_a.log" 2>&1 &
lane_a_pid=$!
sleep 8
gpu0_lane_b > "$log_dir/gpu0_lane_b.log" 2>&1 &
lane_b_pid=$!
echo "$lane_a_pid" > "$log_dir/gpu0_lane_a.pid"
echo "$lane_b_pid" > "$log_dir/gpu0_lane_b.pid"
echo "Started two parallel cuda:0 lanes: lane_a pid=$lane_a_pid, lane_b pid=$lane_b_pid"
wait "$lane_a_pid" "$lane_b_pid"
