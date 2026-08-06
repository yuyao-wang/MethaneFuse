#!/usr/bin/env bash
set -euo pipefail

# Launch immutable 64-row, batch-64 smokes on the formal physical GPU0.
# GPU1 is reserved by another task. Dry-run is the default.

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
runner="$repo_root/research/pretraining_20260727/l89_ragged_cls_experiment.py"
weights="$repo_root/weights/panopticon_vitb14_teacher.pth"
prep_root="/diniuvol/yuyao/methanefuse_research_20260728/l89_clean_replicate_v1"
manifest_root="$prep_root/smoke/manifests"
train_output="$prep_root/smoke/base_train_64_b64.pt"
dev_output="$prep_root/smoke/base_val_64_b64.pt"
train_unit="tempo-l89-clean-smoke-train-b64-v1"
dev_unit="tempo-l89-clean-smoke-dev-b64-v1"

execute="none"
if [[ "${1:-}" == "--run" ]]; then
  execute="both"
  shift
elif [[ "${1:-}" == "--run-dev" ]]; then
  execute="dev"
  shift
fi
if [[ "$#" -ne 0 ]]; then
  printf 'Unexpected arguments: %s\n' "$*" >&2
  exit 2
fi

manifest_sha() {
  sha256sum "$1" | awk '{print $1}'
}
if [[ "$(manifest_sha "$manifest_root/train.csv")" != \
      "44bf8bd06e4f66ad59e32b25e7a679d98f7805d9ea9ed82bc630f84840d0c1a0" ]]; then
  printf 'Train smoke manifest SHA mismatch.\n' >&2
  exit 3
fi
if [[ "$(manifest_sha "$manifest_root/dev.csv")" != \
      "a005f1b047840cb24cef9841206a98b9fc62f94d7f0223d679c2a3609a99ec20" ]]; then
  printf 'Development smoke manifest SHA mismatch.\n' >&2
  exit 4
fi

cache_tail=(
  --weights "$weights"
  --batch-size 64
  --num-workers 6
  --prefetch-factor 1
  --persistent-workers
  --device cuda:0
  --amp-dtype bfloat16
  --storage-dtype float16
  --local-cache-mode off
  --max-rows 64
  --row-selection-seed 20260728
  --max-invalid-t0 0
  --max-read-errors 0
  --log-interval 1
)
systemd_prefix=(
  systemd-run
  --user
  --property=Type=exec
  --property=WorkingDirectory="$repo_root"
  --property=MemoryMax=24G
  --property=CPUQuota=800%
  --setenv=PYTHONPATH="$repo_root"
)
train_command=(
  "${systemd_prefix[@]}"
  --setenv=CUDA_VISIBLE_DEVICES=0
  --unit="$train_unit"
  --description="L89 clean train 64-row batch-64 GPU0 smoke"
  "$python_bin" -B "$runner" cache
  --csv "$manifest_root/train.csv"
  --split train
  --output-cache "$train_output"
  "${cache_tail[@]}"
)
dev_command=(
  "${systemd_prefix[@]}"
  --setenv=CUDA_VISIBLE_DEVICES=0
  --unit="$dev_unit"
  --description="L89 clean dev 64-row batch-64 GPU0 smoke"
  "$python_bin" -B "$runner" cache
  --csv "$manifest_root/dev.csv"
  --split val
  --output-cache "$dev_output"
  "${cache_tail[@]}"
)

print_command() {
  printf '%q ' "$@"
  printf '\n'
}
if [[ "$execute" == "none" ]]; then
  printf 'Dry run only. Exact train unit command:\n'
  print_command "${train_command[@]}"
  printf 'Exact development unit command:\n'
  print_command "${dev_command[@]}"
  exit 0
fi

if [[ "$execute" == "both" ]]; then
  if [[ -e "$train_output" || -e "$dev_output" ]]; then
    printf 'Refusing to overwrite a prior batch-64 smoke output.\n' >&2
    exit 5
  fi
  selected_units=("$train_unit" "$dev_unit")
  selected_gpus=(0)
elif [[ "$execute" == "dev" ]]; then
  if [[ -e "$dev_output" ]]; then
    printf 'Refusing to overwrite the prior development smoke output.\n' >&2
    exit 5
  fi
  selected_units=("$dev_unit")
  selected_gpus=(0)
fi
for unit in "${selected_units[@]}"; do
  if systemctl --user is-active --quiet "$unit"; then
    printf 'Clean smoke unit is already active: %s\n' "$unit" >&2
    exit 6
  fi
done
gpu_snapshot="$(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
)"
for gpu in "${selected_gpus[@]}"; do
  gpu_free_mib="$(
    awk -F, -v target="$gpu" \
      '$1 + 0 == target {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2 + 0}' \
      <<<"$gpu_snapshot"
  )"
  if [[ -z "$gpu_free_mib" || "$gpu_free_mib" -lt 35840 ]]; then
    printf 'GPU%s needs at least 35840 MiB free; observed %s.\n' \
      "$gpu" "${gpu_free_mib:-unavailable}" >&2
    exit 7
  fi
done
if [[ "$execute" == "both" ]]; then
  "${train_command[@]}"
  "${dev_command[@]}"
  printf 'Started user units %s and %s.\n' "$train_unit" "$dev_unit"
else
  "${dev_command[@]}"
  printf 'Started user unit %s.\n' "$dev_unit"
fi
printf 'Monitor: systemctl --user status %s\n' "${selected_units[*]}"
