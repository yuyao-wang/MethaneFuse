#!/usr/bin/env bash
set -u

repo_root="/home/yuyao/panopticon"
log_dir="$repo_root/logs/l89_ablation_20260722"
status_log="$log_dir/hourly_status.log"
runs=(
  l89_drop3_full_bands
  l89_b6_b7
  l89_b1_b5
  l89_non_rgb_b1_b5_b6_b7
)

while true; do
  {
    echo "===== $(date -u +%FT%TZ) ====="
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    for run_name in "${runs[@]}"; do
      run_log="$log_dir/${run_name}.log"
      echo "[$run_name]"
      if [[ -f "$run_log" ]]; then
        rg '^Epoch [0-9]+:' "$run_log" | tail -n 5
        rg '^Epoch [0-9]+ step ' "$run_log" | tail -n 1
      else
        echo "not started"
      fi
    done
    echo
  } >> "$status_log" 2>&1
  sleep 3600
done
