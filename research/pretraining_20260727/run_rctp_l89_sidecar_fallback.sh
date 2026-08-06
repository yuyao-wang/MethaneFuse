#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default.  This launcher is authorized only after an explicit GPU
# handoff.  On --run it snapshots all executable sources, then re-executes the
# frozen launcher so later edits cannot alter a running campaign.

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
artifact_root="/diniuvol/yuyao/methanefuse_research_20260727"
manifest_root="$artifact_root/manifests_staged/l89_6time"
base_cache_root="$artifact_root/cache/l89_ragged_cls_v1"
output_root="${SIDECAR_OUTPUT_ROOT:-$artifact_root/rctp_l89_sidecar_fallback_v1}"
physical_gpu="${SIDECAR_GPU:-1}"
seed=20260728
runner="${SIDECAR_SNAPSHOT_RUNNER:-$repo_root/research/pretraining_20260727/rctp_l89_sidecar_fallback.py}"
evaluator="${SIDECAR_SNAPSHOT_EVALUATOR:-$repo_root/research/pretraining_20260727/rctp_l89_event_balanced_head_followup.py}"
launcher="$repo_root/research/pretraining_20260727/run_rctp_l89_sidecar_fallback.sh"
test_source="$repo_root/research/pretraining_20260727/test_rctp_l89_sidecar_fallback_cpu.py"
protocol_source="$repo_root/research/pretraining_20260727/SIDECAR_RCTP_FALLBACK_PROTOCOL.md"
base_weights="$repo_root/weights/panopticon_vitb14_teacher.pth"
log_root="$output_root/logs"
provenance_root="$output_root/provenance"

execute=0
if [[ "${1:-}" == "--run" ]]; then
  execute=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  printf 'Unexpected arguments: %s\n' "$*" >&2
  exit 2
fi

lower_output="${output_root,,}"
if [[ "$lower_output" =~ (^|[._/-])(test|sealed|holdout)([._/-]|$) ]]; then
  printf 'Refusing forbidden train/dev output path: %s\n' "$output_root" >&2
  exit 3
fi

p4_dir="$output_root/pretrain/p4"
p5_dir="$output_root/pretrain/p5"
cache_root="$output_root/cache"
head_root="$output_root/downstream_event_balanced_seed${seed}"

pretrain_command() {
  local arm="$1"
  local output="$2"
  local -n result="$3"
  result=(
    timeout 15m
    env
    "PYTHONPATH=$repo_root"
    "CUDA_VISIBLE_DEVICES=$physical_gpu"
    "$python_bin" -B "$runner" pretrain
    --arm "$arm"
    --train-csv "$manifest_root/train.csv"
    --dev-csv "$manifest_root/val.csv"
    --train-cache "$base_cache_root/train.pt"
    --dev-cache "$base_cache_root/val.pt"
    --base-weights "$base_weights"
    --output-dir "$output"
    --epochs 2
    --max-train-rows 2048
    --max-dev-rows 512
    --rank 8
    --residual-scale 0.1
    --clean-anchor-weight 0.1
    --batch-size 4
    --eval-batch-size 8
    --num-workers 4
    --prefetch-factor 1
    --learning-rate 1e-4
    --amp-dtype bfloat16
    --seed "$seed"
    --device cuda:0
    --min-cuda-free-gib 30
    --min-runtime-cuda-free-gib 25
    --max-cuda-allocated-gib 12
  )
}

cache_command() {
  local arm="$1"
  local split="$2"
  local sidecar_checkpoint="$3"
  local -n result="$4"
  result=(
    env "PYTHONPATH=$repo_root"
    "$python_bin" -B "$runner" build-cache
    --arm "$arm"
    --split "$split"
    --base-cache "$base_cache_root/$split.pt"
    --output-cache "$cache_root/$arm/$split.pt"
    --batch-size 512
  )
  if [[ -n "$sidecar_checkpoint" ]]; then
    result+=(--sidecar-checkpoint "$sidecar_checkpoint")
  fi
}

pretrain_command p4_response_scrambled "$p4_dir" cmd_p4
pretrain_command p5_correct_response "$p5_dir" cmd_p5
cache_command p0 train "" cmd_p0_train
cache_command p0 val "" cmd_p0_val
cache_command p4 train "$p4_dir/sidecar_best_dev_ap.pt" cmd_p4_train
cache_command p4 val "$p4_dir/sidecar_best_dev_ap.pt" cmd_p4_val
cache_command p5 train "$p5_dir/sidecar_best_dev_ap.pt" cmd_p5_train
cache_command p5 val "$p5_dir/sidecar_best_dev_ap.pt" cmd_p5_val
cmd_heads=(
  env "PYTHONPATH=$repo_root"
  "$python_bin" -B "$evaluator"
  --p0-train-cache "$cache_root/p0/train.pt"
  --p0-val-cache "$cache_root/p0/val.pt"
  --p4-train-cache "$cache_root/p4/train.pt"
  --p4-val-cache "$cache_root/p4/val.pt"
  --p5-train-cache "$cache_root/p5/train.pt"
  --p5-val-cache "$cache_root/p5/val.pt"
  --output-dir "$head_root"
  --epochs 3
  --batch-size 256
  --eval-batch-size 512
  --learning-rate 3e-4
  --weight-decay 0.05
  --model-dim 256
  --num-heads 8
  --seed "$seed"
  --num-threads 12
  --bootstrap-replicates 2000
)

print_command() {
  local label="$1"
  shift
  printf '%-18s' "$label"
  printf ' %q' "$@"
  printf '\n'
}

if [[ "$execute" -eq 0 ]]; then
  printf 'Dry run only; no output or GPU work was started.\n'
  printf 'Frozen contract: residual-only pretext; rank=8, downstream scale=0.1, anchor=0.1, 2 epochs/1024 steps per arm.\n'
  printf 'GPU: physical %s, sequential P4/P5, preflight >=30 GiB free, runtime >=25 GiB free, allocation cap 12 GiB.\n' "$physical_gpu"
  print_command p4-pretrain "${cmd_p4[@]}"
  print_command p5-pretrain "${cmd_p5[@]}"
  print_command p0-train-cache "${cmd_p0_train[@]}"
  print_command p0-val-cache "${cmd_p0_val[@]}"
  print_command p4-train-cache "${cmd_p4_train[@]}"
  print_command p4-val-cache "${cmd_p4_val[@]}"
  print_command p5-train-cache "${cmd_p5_train[@]}"
  print_command p5-val-cache "${cmd_p5_val[@]}"
  print_command event-heads "${cmd_heads[@]}"
  printf 'Estimated GPU use: 3-6 min/arm (6-12 GPU-min total); hard cap 15 min/arm.\n'
  printf 'Estimated peak allocation: 4-8 GiB; enforced cap: 12 GiB.\n'
  exit 0
fi

if [[ "${SIDECAR_FROZEN_REEXEC:-0}" != "1" ]]; then
  if [[ -e "$output_root" ]]; then
    printf 'Refusing to overwrite existing fallback root: %s\n' "$output_root" >&2
    exit 4
  fi
  mkdir -p "$provenance_root"
  cp "$runner" "$provenance_root/rctp_l89_sidecar_fallback.py"
  cp "$evaluator" "$provenance_root/rctp_l89_event_balanced_head_followup.py"
  cp "$launcher" "$provenance_root/run_rctp_l89_sidecar_fallback.sh"
  cp "$test_source" "$provenance_root/test_rctp_l89_sidecar_fallback_cpu.py"
  cp "$protocol_source" "$provenance_root/SIDECAR_RCTP_FALLBACK_PROTOCOL.md"
  sha256sum \
    "$provenance_root/rctp_l89_sidecar_fallback.py" \
    "$provenance_root/rctp_l89_event_balanced_head_followup.py" \
    "$provenance_root/run_rctp_l89_sidecar_fallback.sh" \
    "$provenance_root/test_rctp_l89_sidecar_fallback_cpu.py" \
    "$provenance_root/SIDECAR_RCTP_FALLBACK_PROTOCOL.md" \
    > "$provenance_root/SOURCE_SHA256SUMS.txt"
  exec env \
    SIDECAR_FROZEN_REEXEC=1 \
    SIDECAR_OUTPUT_ROOT="$output_root" \
    SIDECAR_GPU="$physical_gpu" \
    SIDECAR_SNAPSHOT_RUNNER="$provenance_root/rctp_l89_sidecar_fallback.py" \
    SIDECAR_SNAPSHOT_EVALUATOR="$provenance_root/rctp_l89_event_balanced_head_followup.py" \
    bash "$provenance_root/run_rctp_l89_sidecar_fallback.sh" --run
fi

mkdir -p "$log_root" "$cache_root/p0" "$cache_root/p4" "$cache_root/p5"
gpu_line="$(
  nvidia-smi --query-gpu=index,memory.used,memory.free \
    --format=csv,noheader,nounits \
    | awk -F, -v target="$physical_gpu" '$1 + 0 == target {print}'
)"
if [[ -z "$gpu_line" ]]; then
  printf 'Could not resolve physical GPU %s.\n' "$physical_gpu" >&2
  exit 5
fi
gpu_free_mib="$(awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3 + 0}' <<<"$gpu_line")"
if (( gpu_free_mib < 30720 )); then
  printf 'Refusing GPU %s: only %s MiB free, need 30720 MiB.\n' \
    "$physical_gpu" "$gpu_free_mib" >&2
  exit 6
fi
host_available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( host_available_kib < 20971520 )); then
  printf 'Refusing start: host MemAvailable is below 20 GiB.\n' >&2
  exit 7
fi

{
  printf 'started_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf 'physical_gpu=%s\n' "$physical_gpu"
  printf 'gpu_state=%s\n' "$gpu_line"
  printf 'host_mem_available_kib=%s\n' "$host_available_kib"
  sha256sum "$base_weights" "$base_cache_root/train.pt" "$base_cache_root/val.pt"
} > "$log_root/preflight.log"

"${cmd_p4[@]}" 2>&1 | tee "$log_root/p4_pretrain.log"
"${cmd_p5[@]}" 2>&1 | tee "$log_root/p5_pretrain.log"
"${cmd_p0_train[@]}" 2>&1 | tee "$log_root/p0_train_cache.log"
"${cmd_p0_val[@]}" 2>&1 | tee "$log_root/p0_val_cache.log"
"${cmd_p4_train[@]}" 2>&1 | tee "$log_root/p4_train_cache.log"
"${cmd_p4_val[@]}" 2>&1 | tee "$log_root/p4_val_cache.log"
"${cmd_p5_train[@]}" 2>&1 | tee "$log_root/p5_train_cache.log"
"${cmd_p5_val[@]}" 2>&1 | tee "$log_root/p5_val_cache.log"
"${cmd_heads[@]}" 2>&1 | tee "$log_root/event_balanced_heads.log"

sha256sum -c "$provenance_root/SOURCE_SHA256SUMS.txt"
sha256sum \
  "$p4_dir/sidecar_best_dev_ap.pt" \
  "$p4_dir/summary.json" \
  "$p5_dir/sidecar_best_dev_ap.pt" \
  "$p5_dir/summary.json" \
  "$cache_root/p0/train.pt" \
  "$cache_root/p0/val.pt" \
  "$cache_root/p4/train.pt" \
  "$cache_root/p4/val.pt" \
  "$cache_root/p5/train.pt" \
  "$cache_root/p5/val.pt" \
  "$head_root/comparison.json" \
  "$head_root/COMPARISON.md" \
  > "$output_root/ARTIFACT_SHA256SUMS.txt"
printf 'Sidecar-RCTP fallback complete: %s\n' "$output_root"
