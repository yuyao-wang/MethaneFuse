#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yuyao/panopticon"
log_dir="$repo_root/logs/l89_ablation_20260722"
b1_log="$log_dir/l89_b1_b5.log"
b1_checkpoint="$repo_root/checkpoints/l89_ablation_20260722/l89_b1_b5/ckpt_latest.pth"
eval_checkpoint="$repo_root/checkpoints/l89_ablation_20260722/l89_drop3_full_bands/ckpt_best_test.pth"
source_dir="/mnt/engg-niulab/Yuyao/preprocessed_512/L89/l89_6time_temporal_16_resized_to_224"
drop_dir="$repo_root/Upgraded_dataset/l89_6time_temporal_drop_3_hard_events"
eval_log="$log_dir/l89_drop3_same_checkpoint_eval.log"
initial_mtime="$(stat -c %Y "$b1_checkpoint")"

echo "[$(date -u +%FT%TZ)] waiting for B1-5 Epoch 3 checkpoint" | tee -a "$eval_log"
while true; do
  if rg -q '^Epoch 3:' "$b1_log" \
    && [[ "$(stat -c %Y "$b1_checkpoint")" -gt "$initial_mtime" ]]; then
    break
  fi
  sleep 30
done

sleep 60
tmux kill-session -t l89_b1_b5_cuda1 || true
echo "[$(date -u +%FT%TZ)] B1-5 stopped; starting same-checkpoint evaluations" | tee -a "$eval_log"

common_args=(
  --checkpoint "$eval_checkpoint"
  --train_csv "$source_dir/L89_temporal_train.csv"
  --batch_size 8
  --num_workers 8
  --device cuda:1
  --progress_every 100
  --local_cache_mode sync
  --local_cache_dir /diniuvol/yuyao/l89_temporal_cache
  --local_cache_min_free_gb 200
)

"/home/yuyao/miniconda3/envs/panopticon/bin/python" \
  "$repo_root/Upgraded_dataset/eval_l89_temporal_checkpoint.py" \
  "${common_args[@]}" \
  --test_csv "$source_dir/L89_temporal_test.csv" \
  --output_json "$drop_dir/eval_same_checkpoint_full_test.json" \
  2>&1 | tee -a "$eval_log"

"/home/yuyao/miniconda3/envs/panopticon/bin/python" \
  "$repo_root/Upgraded_dataset/eval_l89_temporal_checkpoint.py" \
  "${common_args[@]}" \
  --test_csv "$drop_dir/L89_temporal_test_drop_3_hard_events.csv" \
  --output_json "$drop_dir/eval_same_checkpoint_drop3_test.json" \
  2>&1 | tee -a "$eval_log"

echo "[$(date -u +%FT%TZ)] same-checkpoint evaluations complete" | tee -a "$eval_log"
