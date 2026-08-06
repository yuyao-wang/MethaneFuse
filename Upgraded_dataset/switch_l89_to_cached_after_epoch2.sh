#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yuyao/panopticon"
log_dir="$repo_root/logs/l89_ablation_20260722"
checkpoint_dir="$repo_root/checkpoints/l89_ablation_20260722"
source_dir="/mnt/engg-niulab/Yuyao/preprocessed_512/L89/l89_6time_temporal_16_resized_to_224"
drop_dir="$repo_root/Upgraded_dataset/l89_6time_temporal_drop_3_hard_events"
switch_log="$log_dir/cache_switch.log"
runs=(l89_drop3_full_bands l89_b6_b7)

declare -A initial_mtime
for run_name in "${runs[@]}"; do
  checkpoint="$checkpoint_dir/$run_name/ckpt_latest.pth"
  initial_mtime[$run_name]="$(stat -c %Y "$checkpoint")"
done

echo "[$(date -u +%FT%TZ)] waiting for both Epoch 2 checkpoints" >> "$switch_log"
while true; do
  ready=1
  for run_name in "${runs[@]}"; do
    run_log="$log_dir/${run_name}.log"
    checkpoint="$checkpoint_dir/$run_name/ckpt_latest.pth"
    if ! rg -q '^Epoch 2:' "$run_log"; then
      ready=0
      continue
    fi
    if [[ "$(stat -c %Y "$checkpoint")" -le "${initial_mtime[$run_name]}" ]]; then
      ready=0
    fi
  done
  [[ "$ready" -eq 1 ]] && break
  sleep 30
done

# Allow best-checkpoint writes to finish before terminating the old launcher.
sleep 60
echo "[$(date -u +%FT%TZ)] Epoch 2 saved; switching both runs to shared cache" >> "$switch_log"
tmux kill-session -t l89_ablation_20260722 || true

tmux new-session -d -s l89_drop3_cached \
  "cd '$repo_root' && bash Upgraded_dataset/run_l89_cached_resume.sh 0 l89_drop3_full_bands '$drop_dir/L89_temporal_train_drop_3_hard_events.csv' '$drop_dir/L89_temporal_test_drop_3_hard_events.csv' 0,1,2,3,4,5,6"
tmux new-session -d -s l89_b6_b7_cached \
  "cd '$repo_root' && bash Upgraded_dataset/run_l89_cached_resume.sh 0 l89_b6_b7 '$source_dir/L89_temporal_train.csv' '$source_dir/L89_temporal_test.csv' 5,6"

echo "[$(date -u +%FT%TZ)] cached resume sessions started" >> "$switch_log"
