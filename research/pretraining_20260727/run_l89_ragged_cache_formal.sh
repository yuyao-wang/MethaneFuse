#!/usr/bin/env bash
set -u -o pipefail

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
runner="$repo_root/research/pretraining_20260727/l89_ragged_cls_experiment.py"
manifest_root="/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/l89_6time"
cache_root="/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_ragged_cls_v1"
log_root="$repo_root/logs/journal20260727_l89_ragged_formal"
weights="$repo_root/weights/panopticon_vitb14_teacher.pth"

mkdir -p "$cache_root" "$log_root"

run_cache() {
  local gpu="$1"
  local split="$2"
  "$python_bin" -B "$runner" cache \
    --csv "$manifest_root/${split}.csv" \
    --split "$split" \
    --output-cache "$cache_root/${split}.pt" \
    --weights "$weights" \
    --batch-size 24 \
    --num-workers 8 \
    --prefetch-factor 2 \
    --persistent-workers \
    --device "cuda:$gpu" \
    --amp-dtype bfloat16 \
    --storage-dtype float16 \
    --local-cache-mode off \
    --max-invalid-t0 0 \
    --max-read-errors 0 \
    --log-interval 25
}

run_cache 0 train 2>&1 | tee "$log_root/train.log" &
train_pid=$!
printf '%s\n' "$train_pid" > "$log_root/train.pid"

run_cache 1 val 2>&1 | tee "$log_root/val.log" &
val_pid=$!
printf '%s\n' "$val_pid" > "$log_root/val.pid"

status=0
if ! wait "$train_pid"; then
  status=1
fi
if ! wait "$val_pid"; then
  status=1
fi
exit "$status"
