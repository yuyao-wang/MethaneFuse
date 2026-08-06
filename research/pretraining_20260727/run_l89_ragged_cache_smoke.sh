#!/usr/bin/env bash
set -u -o pipefail

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
runner="$repo_root/research/pretraining_20260727/l89_ragged_cls_experiment.py"
manifest_root="/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/l89_6time"
cache_root="/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_ragged_cls_smoke_v1"
log_root="$repo_root/logs/journal20260727_l89_ragged_smoke"
weights="$repo_root/weights/panopticon_vitb14_teacher.pth"

mkdir -p "$cache_root" "$log_root"

run_cache() {
  local gpu="$1"
  local split="$2"
  "$python_bin" -B "$runner" cache \
    --csv "$manifest_root/${split}.csv" \
    --split "$split" \
    --output-cache "$cache_root/${split}64.pt" \
    --weights "$weights" \
    --max-rows 64 \
    --batch-size 8 \
    --num-workers 4 \
    --device "cuda:$gpu" \
    --local-cache-mode off \
    --log-interval 1
}

run_cache 0 train 2>&1 | tee "$log_root/train64.log" &
train_pid=$!
run_cache 1 val 2>&1 | tee "$log_root/val64.log" &
val_pid=$!

status=0
if ! wait "$train_pid"; then
  status=1
fi
if ! wait "$val_pid"; then
  status=1
fi
exit "$status"
