#!/usr/bin/env bash
set -u -o pipefail

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
runner="$repo_root/Upgraded_dataset/dino_classifier_head_l89_temporal_satmae.py"
manifest_root="/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/l89_3time"
output_root="/diniuvol/yuyao/methanefuse_research_20260727/panopticon_l89_input_screen_20260727"
log_root="$repo_root/logs/journal20260727_l89_input_screen"

mkdir -p "$log_root"

run_arm() {
  local gpu="$1"
  local name="$2"
  local mode="$3"
  local slots="$4"
  local log_path="$log_root/${name}.log"

  {
    echo "[$(date -u +%FT%TZ)] start name=$name gpu=$gpu mode=$mode slots=$slots"
    "$python_bin" -B "$runner" \
      --train_csv "$manifest_root/train.csv" \
      --test_csv "$manifest_root/val.csv" \
      --weights "$repo_root/weights/panopticon_vitb14_teacher.pth" \
      --path_columns path_t0,path_prev1,path_seasonal \
      --time_columns t0_image_time,prev1_image_time,seasonal_image_time \
      --input_mode "$mode" \
      --residual_slots "$slots" \
      --use_precomputed_stats \
      --batch_size 13 \
      --epochs 1 \
      --head_lr 1e-3 \
      --temporal_lr 1e-3 \
      --lr_scheduler none \
      --num_workers 2 \
      --prefetch_factor 2 \
      --seed 20260727 \
      --device "cuda:$gpu" \
      --log_interval 100 \
      --local_cache_mode off \
      --checkpoint_dir "$output_root" \
      --run_name "$name"
    status=$?
    echo "[$(date -u +%FT%TZ)] finish name=$name status=$status"
    return "$status"
  } 2>&1 | tee "$log_path"
}

run_arm 0 l89_panopticon_t0 current path_prev1,path_seasonal &
pid_t0=$!
run_arm 0 l89_panopticon_raw3 raw path_prev1,path_seasonal &
pid_raw3=$!
run_arm 1 l89_panopticon_residual2 residual path_prev1,path_seasonal &
pid_residual2=$!
run_arm 1 l89_panopticon_t0_residual2 current_residual path_prev1,path_seasonal &
pid_t0_residual2=$!

printf '%s\n' "$pid_t0" > "$log_root/l89_panopticon_t0.pid"
printf '%s\n' "$pid_raw3" > "$log_root/l89_panopticon_raw3.pid"
printf '%s\n' "$pid_residual2" > "$log_root/l89_panopticon_residual2.pid"
printf '%s\n' "$pid_t0_residual2" > "$log_root/l89_panopticon_t0_residual2.pid"

status=0
for pid in "$pid_t0" "$pid_raw3" "$pid_residual2" "$pid_t0_residual2"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
