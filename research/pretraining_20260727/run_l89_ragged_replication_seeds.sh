#!/usr/bin/env bash
set -u -o pipefail

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
runner="$repo_root/research/pretraining_20260727/l89_ragged_cls_experiment.py"
cache_root="/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_ragged_cls_v1"
result_root="/diniuvol/yuyao/methanefuse_research_20260727/results"
log_root="$repo_root/logs/journal20260727_l89_ragged_replication"

mkdir -p "$log_root"

run_seed() {
  local seed="$1"
  "$python_bin" -B "$runner" train-heads \
    --train-cache "$cache_root/train.pt" \
    --val-cache "$cache_root/val.pt" \
    --output-dir "$result_root/l89_ragged_cls_v1_seed${seed}" \
    --epochs 3 \
    --batch-size 512 \
    --eval-batch-size 1024 \
    --learning-rate 3e-4 \
    --weight-decay 0.05 \
    --model-dim 256 \
    --num-heads 8 \
    --dropout 0.1 \
    --seed "$seed" \
    --device cuda:0
}

run_seed 20260728 2>&1 | tee "$log_root/seed20260728.log" &
pid_a=$!
run_seed 20260729 2>&1 | tee "$log_root/seed20260729.log" &
pid_b=$!

status=0
if ! wait "$pid_a"; then
  status=1
fi
if ! wait "$pid_b"; then
  status=1
fi
exit "$status"
