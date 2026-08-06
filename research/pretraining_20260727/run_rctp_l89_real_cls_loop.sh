#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default. Pass --run only after legacy extraction has released both
# GPUs. Hard timeouts cap the two-arm total at 84 GPU-minutes:
#   per arm: pretrain 20m + train-cache 12m + dev-cache 10m = 42m.

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
artifact_root="/diniuvol/yuyao/methanefuse_research_20260727"
manifest_root="$artifact_root/manifests_staged/l89_6time"
base_cache_root="$artifact_root/cache/l89_ragged_cls_v1"
loop_root="$artifact_root/rctp_l89_real_cls_v1"
pretrain_runner="$repo_root/research/pretraining_20260727/rctp_l89_continue_pretrain.py"
downstream_runner="$repo_root/research/pretraining_20260727/l89_ragged_cls_experiment.py"
audit_runner="$repo_root/research/pretraining_20260727/audit_rctp_l89_real_cls_loop.py"
base_weights="$repo_root/weights/panopticon_vitb14_teacher.pth"
seed="${RCTP_SEED:-20260728}"
epochs="${RCTP_PRETRAIN_EPOCHS:-2}"
train_rows="${RCTP_TRAIN_ROWS:-2048}"
dev_rows="${RCTP_DEV_ROWS:-512}"
gpu_p4="${RCTP_GPU_P4:-0}"
gpu_p5="${RCTP_GPU_P5:-1}"
train_last_blocks="${RCTP_TRAIN_LAST_BLOCKS:-2}"
backbone_lr="${RCTP_BACKBONE_LR:-1e-5}"
clean_anchor_weight="${RCTP_CLEAN_ANCHOR_WEIGHT:-0}"
clean_anchor_metric="${RCTP_CLEAN_ANCHOR_METRIC:-cosine}"
run_tag="${RCTP_RUN_TAG:-}"
if [[ -n "$run_tag" ]]; then
  loop_root="${loop_root}_${run_tag}"
fi

execute=0
if [[ "${1:-}" == "--run" ]]; then
  execute=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  printf 'Unexpected arguments: %s\n' "$*" >&2
  exit 2
fi

p4_pretrain="$loop_root/pretrain/p4"
p5_pretrain="$loop_root/pretrain/p5"
p4_cache="$loop_root/cache/p4"
p5_cache="$loop_root/cache/p5"
head_root="$loop_root/downstream_role_only_seed${seed}"
log_root="$repo_root/logs/rctp_l89_real_cls_v1"

pretrain_command() {
  local arm="$1"
  local gpu="$2"
  local output="$3"
  local -n result="$4"
  result=(
    timeout 20m
    env "CUDA_VISIBLE_DEVICES=$gpu"
    "$python_bin" -B "$pretrain_runner"
    --arm "$arm"
    --train-csv "$manifest_root/train.csv"
    --dev-csv "$manifest_root/val.csv"
    --train-cache "$base_cache_root/train.pt"
    --dev-cache "$base_cache_root/val.pt"
    --base-weights "$base_weights"
    --output-dir "$output"
    --epochs "$epochs"
    --max-train-rows "$train_rows"
    --max-dev-rows "$dev_rows"
    --train-last-blocks "$train_last_blocks"
    --batch-size 4
    --eval-batch-size 8
    --num-workers 6
    --prefetch-factor 2
    --backbone-lr "$backbone_lr"
    --probe-lr 1e-4
    --clean-anchor-weight "$clean_anchor_weight"
    --clean-anchor-metric "$clean_anchor_metric"
    --amp-dtype bfloat16
    --seed "$seed"
    --device cuda:0
  )
}

cache_command() {
  local arm="$1"
  local split="$2"
  local gpu="$3"
  local weights="$4"
  local cache_root="$5"
  local timeout_value="$6"
  local -n result="$7"
  result=(
    timeout "$timeout_value"
    env "CUDA_VISIBLE_DEVICES=$gpu"
    "$python_bin" -B "$downstream_runner" cache
    --csv "$manifest_root/${split}.csv"
    --split "$split"
    --output-cache "$cache_root/${split}.pt"
    --weights "$weights"
    --batch-size 24
    --num-workers 8
    --prefetch-factor 2
    --persistent-workers
    --device cuda:0
    --amp-dtype bfloat16
    --storage-dtype float16
    --local-cache-mode off
    --max-invalid-t0 0
    --max-read-errors 0
    --log-interval 25
  )
}

head_command() {
  local arm="$1"
  local train_cache="$2"
  local val_cache="$3"
  local -n result="$4"
  result=(
    "$python_bin" -B "$downstream_runner" train-heads
    --train-cache "$train_cache"
    --val-cache "$val_cache"
    --output-dir "$head_root/$arm"
    --arms role_only
    --epochs 3
    --batch-size 256
    --eval-batch-size 512
    --learning-rate 3e-4
    --weight-decay 0.05
    --model-dim 256
    --num-heads 8
    --seed "$seed"
    --device cpu
  )
}

print_command() {
  local label="$1"
  shift
  printf '%-20s' "$label"
  printf ' %q' "$@"
  printf '\n'
}

pretrain_command p4_response_scrambled "$gpu_p4" "$p4_pretrain" cmd_p4_pretrain
pretrain_command p5_correct_response "$gpu_p5" "$p5_pretrain" cmd_p5_pretrain
cache_command p4 train "$gpu_p4" "$p4_pretrain/backbone_best_dev_ap.pth" "$p4_cache" 12m cmd_p4_train_cache
cache_command p4 val "$gpu_p4" "$p4_pretrain/backbone_best_dev_ap.pth" "$p4_cache" 10m cmd_p4_val_cache
cache_command p5 train "$gpu_p5" "$p5_pretrain/backbone_best_dev_ap.pth" "$p5_cache" 12m cmd_p5_train_cache
cache_command p5 val "$gpu_p5" "$p5_pretrain/backbone_best_dev_ap.pth" "$p5_cache" 10m cmd_p5_val_cache
head_command p0 "$base_cache_root/train.pt" "$base_cache_root/val.pt" cmd_p0_head
head_command p4 "$p4_cache/train.pt" "$p4_cache/val.pt" cmd_p4_head
head_command p5 "$p5_cache/train.pt" "$p5_cache/val.pt" cmd_p5_head
cmd_audit=(
  "$python_bin" -B "$audit_runner"
  --p0-dir "$head_root/p0"
  --p4-dir "$head_root/p4"
  --p5-dir "$head_root/p5"
  --p4-pretrain-dir "$p4_pretrain"
  --p5-pretrain-dir "$p5_pretrain"
  --p0-weights "$base_weights"
  --p4-weights "$p4_pretrain/backbone_best_dev_ap.pth"
  --p5-weights "$p5_pretrain/backbone_best_dev_ap.pth"
  --output-json "$head_root/comparison.json"
  --output-md "$head_root/COMPARISON.md"
)

if [[ "$execute" -eq 0 ]]; then
  printf 'Dry run only; no GPU work started. Planned matched loop:\n'
  printf 'Config: last_blocks=%s backbone_lr=%s clean_anchor_weight=%s clean_anchor_metric=%s run_tag=%s\n' \
    "$train_last_blocks" "$backbone_lr" "$clean_anchor_weight" \
    "$clean_anchor_metric" "${run_tag:-<default>}"
  print_command p4-pretrain "${cmd_p4_pretrain[@]}"
  print_command p5-pretrain "${cmd_p5_pretrain[@]}"
  print_command p4-cache-train "${cmd_p4_train_cache[@]}"
  print_command p4-cache-val "${cmd_p4_val_cache[@]}"
  print_command p5-cache-train "${cmd_p5_train_cache[@]}"
  print_command p5-cache-val "${cmd_p5_val_cache[@]}"
  print_command p0-head "${cmd_p0_head[@]}"
  print_command p4-head "${cmd_p4_head[@]}"
  print_command p5-head "${cmd_p5_head[@]}"
  print_command audit "${cmd_audit[@]}"
  printf '\nHard GPU budget: 2 arms x (20m + 12m + 10m) = 84 GPU-min maximum.\n'
  exit 0
fi

if [[ "${RCTP_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  active_compute="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]'
  )"
  if [[ -n "$active_compute" ]]; then
    printf 'Refusing to start: GPU compute processes are active. Wait for legacy extraction or set RCTP_ALLOW_BUSY_GPU=1 explicitly.\n' >&2
    exit 3
  fi
fi

mkdir -p "$loop_root/pretrain" "$loop_root/cache" "$head_root" "$log_root"

"${cmd_p4_pretrain[@]}" 2>&1 | tee "$log_root/p4_pretrain.log" &
pid_p4=$!
"${cmd_p5_pretrain[@]}" 2>&1 | tee "$log_root/p5_pretrain.log" &
pid_p5=$!
wait "$pid_p4"
wait "$pid_p5"

(
  mkdir -p "$p4_cache"
  "${cmd_p4_train_cache[@]}" 2>&1 | tee "$log_root/p4_cache_train.log"
  "${cmd_p4_val_cache[@]}" 2>&1 | tee "$log_root/p4_cache_val.log"
) &
pid_p4_cache=$!
(
  mkdir -p "$p5_cache"
  "${cmd_p5_train_cache[@]}" 2>&1 | tee "$log_root/p5_cache_train.log"
  "${cmd_p5_val_cache[@]}" 2>&1 | tee "$log_root/p5_cache_val.log"
) &
pid_p5_cache=$!
wait "$pid_p4_cache"
wait "$pid_p5_cache"

"${cmd_p0_head[@]}" 2>&1 | tee "$log_root/p0_head.log"
"${cmd_p4_head[@]}" 2>&1 | tee "$log_root/p4_head.log"
"${cmd_p5_head[@]}" 2>&1 | tee "$log_root/p5_head.log"
"${cmd_audit[@]}" 2>&1 | tee "$log_root/audit.log"
