#!/usr/bin/env bash
set -euo pipefail

# Bounded clean-inner exploratory D1 screen. Dry-run is the default.
# The script never accesses the formal output root and is pinned to physical
# GPU0. Physical GPU1 remains outside this task.

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
base_root="/tmp/l89_clean_exploratory_base_v1"
output_root="/tmp/l89_clean_d1_exploratory_screen_v1"
train_base="$base_root/train.pt"
dev_base="$base_root/val.pt"
p0_cache_root="$output_root/p0_cache"
p0_head_root="$output_root/p0_head"
p0_checkpoint="$p0_head_root/p0/checkpoint_best_event_balanced_ap.pt"
sidecar_runner="$repo_root/research/pretraining_20260727/rctp_l89_sidecar_fallback.py"
p0_head_runner="$repo_root/research/tempo_20260728/build_l89_clean_exploratory_p0_head.py"
d1_runner="$repo_root/research/tempo_20260728/tempo_l89_global.py"
summarizer="$repo_root/research/tempo_20260728/summarize_l89_clean_d1_exploratory.py"

execute=0
if [[ "${1:-}" == "--run" ]]; then
  execute=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  printf 'Unexpected arguments: %s\n' "$*" >&2
  exit 2
fi

for path in "$base_root" "$output_root"; do
  lower="${path,,}"
  if [[ "$lower" =~ (^|[._/-])(test|sealed|holdout|outer)([._/-]|$) ]]; then
    printf 'Refusing held-out-like path: %s\n' "$path" >&2
    exit 3
  fi
done
if [[ "$output_root" == \
      "/diniuvol/yuyao/methanefuse_research_20260728/l89_clean_replicate_run_v1" ]]; then
  printf 'Refusing the frozen formal output root.\n' >&2
  exit 4
fi

configs=(
  "c0_default 192 8e-4 0.02 0.15"
  "c1_lr4e4 192 4e-4 0.02 0.15"
  "c2_lr12e3 192 1.2e-3 0.02 0.15"
  "c3_dim128 128 8e-4 0.02 0.15"
  "c4_dim256 256 8e-4 0.02 0.15"
  "c5_dropout30 192 8e-4 0.02 0.30"
)

print_d1_command() {
  local name="$1"
  local dim="$2"
  local lr="$3"
  local wd="$4"
  local dropout="$5"
  printf '%q ' \
    env \
    PYTHONPATH="$repo_root" \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    "$python_bin" -B "$d1_runner" run \
    --train-cache "$train_base" \
    --dev-cache "$dev_base" \
    --base-kind event_balanced_p0 \
    --event-base-checkpoint "$p0_checkpoint" \
    --output-dir "$output_root/screen/$name" \
    --arms p0_base,d1_gated_delta \
    --seeds 20260728 \
    --epochs 4 \
    --patience 1 \
    --batch-size 512 \
    --eval-batch-size 1024 \
    --temporal-dim "$dim" \
    --learning-rate "$lr" \
    --weight-decay "$wd" \
    --dropout "$dropout" \
    --device cuda:0
  printf '\n'
}

if (( ! execute )); then
  printf 'DRY RUN: clean-inner D1 exploratory sidecar\n'
  printf 'base train: %s\nbase dev: %s\noutput: %s\n' \
    "$train_base" "$dev_base" "$output_root"
  printf 'P0 build uses existing sidecar cache builder and an imported exact P0 head.\n'
  for row in "${configs[@]}"; do
    read -r name dim lr wd dropout <<<"$row"
    print_d1_command "$name" "$dim" "$lr" "$wd" "$dropout"
  done
  printf 'Then freeze AP-only selection and run seeds 20260727/28/29.\n'
  exit 0
fi

for path in "$train_base" "$dev_base"; do
  if [[ ! -f "$path" ]]; then
    printf 'Base cache is not complete: %s\n' "$path" >&2
    exit 5
  fi
done
if [[ -e "$output_root" ]]; then
  printf 'Refusing to overwrite exploratory output: %s\n' "$output_root" >&2
  exit 6
fi

# This check is deliberately adjacent to the GPU stage. Two extractors measured
# 44.7 GiB; the D1 screen starts only after GPU0 has at least 64 GiB free.
gpu0_free="$(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, '$1+0==0 {gsub(/ /,"",$2); print $2}'
)"
if [[ -z "$gpu0_free" || "$gpu0_free" -lt 65536 ]]; then
  printf 'GPU0 is not ready for the screen: free=%s MiB\n' "$gpu0_free" >&2
  exit 7
fi

mkdir -p "$output_root/logs" "$p0_cache_root"
sha256sum \
  "$train_base" "$dev_base" "$sidecar_runner" "$p0_head_runner" \
  "$d1_runner" "$summarizer" >"$output_root/INPUT_SHA256SUMS.txt"
{
  printf 'started_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf 'physical_gpu=0\n'
  printf 'gpu1_used=false\n'
  printf 'gpu0_free_mib=%s\n' "$gpu0_free"
  printf 'screen_configs=6\n'
  printf 'screen_seed=20260728\n'
  printf 'epochs_per_config=4\n'
  printf 'patience=1\n'
  printf 'selection=event_balanced_ap_only\n'
  printf 'ap_tie_band=0.0005\n'
  printf 'test_or_sealed_or_holdout_or_outer_read=false\n'
} >"$output_root/PREFLIGHT.txt"

env PYTHONPATH="$repo_root" CUDA_VISIBLE_DEVICES="" \
  "$python_bin" -B "$sidecar_runner" build-cache \
  --arm p0 --split train --base-cache "$train_base" \
  --output-cache "$p0_cache_root/train.pt" --batch-size 512 \
  >"$output_root/logs/p0_cache_train.log" 2>&1 &
p0_train_pid=$!
env PYTHONPATH="$repo_root" CUDA_VISIBLE_DEVICES="" \
  "$python_bin" -B "$sidecar_runner" build-cache \
  --arm p0 --split val --base-cache "$dev_base" \
  --output-cache "$p0_cache_root/val.pt" --batch-size 512 \
  >"$output_root/logs/p0_cache_val.log" 2>&1 &
p0_val_pid=$!
wait "$p0_train_pid"
wait "$p0_val_pid"

env PYTHONPATH="$repo_root" CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
  "$python_bin" -B "$p0_head_runner" \
  --train-cache "$p0_cache_root/train.pt" \
  --dev-cache "$p0_cache_root/val.pt" \
  --output-dir "$p0_head_root" \
  --epochs 3 --batch-size 256 --eval-batch-size 512 \
  --learning-rate 3e-4 --weight-decay 0.05 \
  --model-dim 256 --num-heads 8 --seed 20260728 --num-threads 12 \
  >"$output_root/logs/p0_head.log" 2>&1

declare -a screen_pids=()
declare -a screen_names=()
for row in "${configs[@]}"; do
  read -r name dim lr wd dropout <<<"$row"
  mkdir -p "$output_root/screen"
  timeout 20m \
    env PYTHONPATH="$repo_root" CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$python_bin" -B "$d1_runner" run \
    --train-cache "$train_base" \
    --dev-cache "$dev_base" \
    --base-kind event_balanced_p0 \
    --event-base-checkpoint "$p0_checkpoint" \
    --output-dir "$output_root/screen/$name" \
    --arms p0_base,d1_gated_delta \
    --seeds 20260728 \
    --epochs 4 --patience 1 \
    --batch-size 512 --eval-batch-size 1024 \
    --temporal-dim "$dim" --learning-rate "$lr" \
    --weight-decay "$wd" --dropout "$dropout" \
    --device cuda:0 \
    >"$output_root/logs/screen_${name}.log" 2>&1 &
  screen_pids+=("$!")
  screen_names+=("$name")
done
screen_failed=0
for index in "${!screen_pids[@]}"; do
  if ! wait "${screen_pids[$index]}"; then
    printf 'Screen failed: %s\n' "${screen_names[$index]}" >&2
    screen_failed=1
  fi
done
if (( screen_failed )); then
  exit 8
fi

env PYTHONPATH="$repo_root" CUDA_VISIBLE_DEVICES="" \
  "$python_bin" -B "$summarizer" select --root "$output_root" \
  >"$output_root/logs/select.log" 2>&1
mapfile -t best_args <"$output_root/BEST_ARGS.txt"
timeout 30m \
  env PYTHONPATH="$repo_root" CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  "$python_bin" -B "$d1_runner" run \
  --train-cache "$train_base" \
  --dev-cache "$dev_base" \
  --base-kind event_balanced_p0 \
  --event-base-checkpoint "$p0_checkpoint" \
  --output-dir "$output_root/three_seed" \
  --arms p0_base,d1_gated_delta \
  --seeds 20260727,20260728,20260729 \
  --epochs 4 --patience 1 \
  --batch-size 512 --eval-batch-size 1024 \
  "${best_args[@]}" --device cuda:0 \
  >"$output_root/logs/three_seed.log" 2>&1

env PYTHONPATH="$repo_root" CUDA_VISIBLE_DEVICES="" \
  "$python_bin" -B "$summarizer" finalize \
  --root "$output_root" --bootstrap-replicates 2000 \
  --bootstrap-seed 2026072817 \
  >"$output_root/logs/finalize.log" 2>&1
sha256sum -c "$output_root/INPUT_SHA256SUMS.txt"
sha256sum \
  "$output_root/SCREEN_SELECTION.json" \
  "$output_root/SCREEN_TABLE.csv" \
  "$output_root/FINAL_RESULT.json" \
  "$output_root/FINAL_RESULT.md" \
  "$output_root/THREE_SEED_LOGIT_ENSEMBLE_PREDICTIONS.csv" \
  >"$output_root/OUTPUT_SHA256SUMS.txt"
printf 'COMPLETE %s\n' "$output_root"
