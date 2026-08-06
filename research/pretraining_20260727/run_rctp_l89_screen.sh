#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
runner="$repo_root/research/pretraining_20260727/rctp_l89_screen.py"
artifact_root="/diniuvol/yuyao/methanefuse_research_20260727"
output_dir="$artifact_root/results/rctp_l89_screen_seed20260728"
log_path="$repo_root/logs/rctp_l89_screen_seed20260728.log"
gpu="${RCTP_GPU:-0}"

execute=0
if [[ "${1:-}" == "--run" ]]; then
  execute=1
  shift
fi

command=(
  env "CUDA_VISIBLE_DEVICES=$gpu"
  "$python_bin" -B "$runner"
  --train-csv "$artifact_root/manifests_staged/l89_6time/train.csv"
  --dev-csv "$artifact_root/manifests_staged/l89_6time/val.csv"
  --train-cache "$artifact_root/cache/l89_ragged_cls_v1/train.pt"
  --dev-cache "$artifact_root/cache/l89_ragged_cls_v1/val.pt"
  --weights "$repo_root/weights/panopticon_vitb14_teacher.pth"
  --output-dir "$output_dir"
  --epochs 3
  --max-train-rows 4096
  --max-dev-rows 2048
  --batch-size 12
  --eval-batch-size 16
  --num-workers 6
  --prefetch-factor 2
  --amp-dtype bfloat16
  --learning-rate 3e-4
  --seed 20260728
  "$@"
)

if [[ "$execute" -eq 0 ]]; then
  printf 'Dry run only. Launch explicitly with:\n'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$output_dir" "$(dirname "$log_path")"
"${command[@]}" 2>&1 | tee "$log_path"
