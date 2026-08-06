#!/usr/bin/env bash
set -euo pipefail

# Small-capacity, low-learning-rate follow-up after the 256-d axial pilot.
# Development only: this launcher has no test input or sealed-test switch.

mode=plan
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--plan|--run]" >&2
  exit 2
fi
case "${1:---plan}" in
  --plan) ;;
  --run) mode=run ;;
  *) echo "Usage: $0 [--plan|--run]" >&2; exit 2 ;;
esac

repo=${REPO_ROOT:-/home/yuyao/panopticon}
python_bin=${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}
formal_root=${FORMAL_ROOT:-/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5}
train_cache=${TRAIN_CACHE:-${formal_root}/features/pilot16k_train_core_universal_s2hybrid.pt}
dev_cache=${DEV_CACHE:-${formal_root}/features/dev_universal_s2hybrid.pt}
output_root=${HEADS_ROOT:-${formal_root}/pilot16k_heads_compact_axial}
runner=${repo}/research/pretraining_20260727/query360_two_axis_full_legacy.py
gpu=${PILOT_GPU:-1}

if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "PILOT_GPU must be a non-negative integer; got: $gpu" >&2
  exit 2
fi
for value in "$train_cache" "$dev_cache" "$output_root"; do
  lower=${value,,}
  if [[ "$lower" == *sealed* ]] || [[ "$lower" =~ (^|[/_.-])test([/_.-]|$) ]]; then
    echo "Refusing test/sealed-like development path: $value" >&2
    exit 2
  fi
done

names=(
  universal_two_axis_d64_lr1e5
  universal_two_axis_d64_lr3e5
  universal_two_axis_d64_lr1e4
  universal_scale_aware_d64_lr1e5
  universal_scale_aware_d64_lr3e5
  universal_scale_aware_d64_lr1e4
)
arms=(
  two_axis_query
  two_axis_query
  two_axis_query
  scale_aware_two_axis_query
  scale_aware_two_axis_query
  scale_aware_two_axis_query
)
lrs=(1e-5 3e-5 1e-4 1e-5 3e-5 1e-4)

printf 'Protocol: dev-only compact axial; epochs=2 model_dim=64 depth=1 dropout=0\n'
printf 'Train: %s\nDev:   %s\nGPU:   %s\n' "$train_cache" "$dev_cache" "$gpu"
printf '%-42s %-30s %-8s\n' RUN ARM LR
for index in "${!names[@]}"; do
  printf '%-42s %-30s %-8s\n' \
    "${names[$index]}" "${arms[$index]}" "${lrs[$index]}"
done
if [[ "$mode" == plan ]]; then
  exit 0
fi

for required in "$python_bin" "$runner" "$train_cache" "$dev_cache"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done
[[ ! -e "$output_root" ]] || {
  echo "Refusing existing output root: $output_root" >&2
  exit 3
}
mkdir -p "$output_root"

pids=()
for index in "${!names[@]}"; do
  run_dir=${output_root}/${names[$index]}
  mkdir "$run_dir"
  (
    "$python_bin" "$runner" train \
      --train-cache "$train_cache" \
      --dev-cache "$dev_cache" \
      --output-dir "$run_dir" \
      --arm "${arms[$index]}" \
      --base-mode universal \
      --epochs 2 \
      --seed 42 \
      --batch-size 1024 \
      --eval-batch-size 4096 \
      --learning-rate "${lrs[$index]}" \
      --weight-decay 0.01 \
      --sensor-aux-weight 0.05 \
      --axis-aux-weight 0.05 \
      --grad-clip 1.0 \
      --model-dim 64 \
      --num-heads 4 \
      --temporal-depth 1 \
      --mlp-ratio 1.0 \
      --dropout 0.0 \
      --selection-metric best_binary_f1 \
      --device "cuda:${gpu}" \
      2>&1 | tee "${run_dir}/stdout_stderr.log"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one compact axial pilot failed." >&2
  exit 1
fi
echo "All 6 compact axial development pilots completed."
