#!/usr/bin/env bash
set -u -o pipefail

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
runner="$repo_root/research/pretraining_20260727/s5p_native_grid_experiment.py"
work_root="/diniuvol/yuyao/methanefuse_research_20260727"
log_root="$repo_root/logs/journal20260727_s5p_native_replication"

mkdir -p "$log_root"

run_seed() {
  local seed="$1"
  "$python_bin" -B "$runner" \
    --train_csv "$work_root/manifests_staged/s5p/train.csv" \
    --val_csv "$work_root/manifests_staged/s5p/val.csv" \
    --local_npz_root "$work_root/cache/s5p_npz" \
    --cache_dir "$work_root/cache/s5p_native_grid_experiment_v2" \
    --output_json "$work_root/results/s5p_native_grid_v2_seed${seed}.json" \
    --device cuda:1 \
    --epochs 3 \
    --seed "$seed" \
    --batch_size 512 \
    --feature_workers 8 \
    --progress_every 0
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
