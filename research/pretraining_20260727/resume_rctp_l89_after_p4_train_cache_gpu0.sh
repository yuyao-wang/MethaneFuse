#!/usr/bin/env bash
set -euo pipefail

# One-shot continuation for the interrupted default RCTP L89 campaign.
#
# The completed P4 pretraining checkpoint and P4 train cache are immutable
# inputs, bound below by exact SHA-256 values.  This script starts at P4 val
# cache and then runs P5 pretraining/caches, the matched CPU heads, and audit.
# It never exposes physical GPU 1.  Dry-run plus artifact audit is the default.

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
log_root="$repo_root/logs/rctp_l89_real_cls_v1"

# Frozen default-run protocol.  These values must remain matched to P4.
seed=20260728
epochs=2
train_rows=2048
dev_rows=512
train_last_blocks=2
backbone_lr=1e-5
clean_anchor_weight=0
clean_anchor_metric=cosine
physical_gpu=0
minimum_start_free_mib=45000

p4_pretrain="$loop_root/pretrain/p4"
p5_pretrain="$loop_root/pretrain/p5"
p4_cache="$loop_root/cache/p4"
p5_cache="$loop_root/cache/p5"
head_root="$loop_root/downstream_role_only_seed${seed}"

p4_weights="$p4_pretrain/backbone_best_dev_ap.pth"
p4_summary="$p4_pretrain/summary.json"
p4_train_cache="$p4_cache/train.pt"
p4_train_audit="$p4_train_cache.json"

expected_p4_weights_sha256="5fba988efc8de5ee1af185603a4ec353969d7a84898b8eda63ea621f28c4a16a"
expected_p4_summary_sha256="411624e1d6c3c0d922c96c5f78152e8a1a25b2667d2bd1ab76daa7de81c4183c"
expected_p4_train_cache_sha256="74c2c61110ec0d6d04a86c1c646654ef8927826d9ffa4ef497f216b55e2109ef"
expected_p4_train_audit_sha256="d6ec58a2dea925572d7c524ac12ae72132b5d3d39ec422b87b9fa0bb6febfa21"

execute=0
if [[ "${1:-}" == "--run" ]]; then
  execute=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  printf 'Unexpected arguments: %s\n' "$*" >&2
  exit 2
fi

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  if [[ ! -f "$path" ]]; then
    printf 'Missing required immutable artifact: %s\n' "$path" >&2
    exit 10
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    printf 'SHA-256 mismatch for %s\nexpected=%s\nactual=%s\n' \
      "$path" "$expected" "$actual" >&2
    exit 11
  fi
  printf 'SHA256_OK %s %s\n' "$actual" "$path"
}

refuse_existing() {
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      printf 'Refusing to overwrite continuation output: %s\n' "$path" >&2
      exit 12
    fi
  done
}

print_command() {
  local label="$1"
  shift
  printf '%-20s' "$label"
  printf ' %q' "$@"
  printf '\n'
}

pretrain_command() {
  local -n result="$1"
  result=(
    timeout 20m
    env "CUDA_VISIBLE_DEVICES=$physical_gpu"
    "$python_bin" -B "$pretrain_runner"
    --arm p5_correct_response
    --train-csv "$manifest_root/train.csv"
    --dev-csv "$manifest_root/val.csv"
    --train-cache "$base_cache_root/train.pt"
    --dev-cache "$base_cache_root/val.pt"
    --base-weights "$base_weights"
    --output-dir "$p5_pretrain"
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
  local split="$1"
  local weights="$2"
  local cache_root="$3"
  local timeout_value="$4"
  local -n result="$5"
  result=(
    timeout "$timeout_value"
    env "CUDA_VISIBLE_DEVICES=$physical_gpu"
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

cache_command val "$p4_weights" "$p4_cache" 10m cmd_p4_val_cache
pretrain_command cmd_p5_pretrain
cache_command train "$p5_pretrain/backbone_best_dev_ap.pth" "$p5_cache" 12m cmd_p5_train_cache
cache_command val "$p5_pretrain/backbone_best_dev_ap.pth" "$p5_cache" 10m cmd_p5_val_cache
head_command p0 "$base_cache_root/train.pt" "$base_cache_root/val.pt" cmd_p0_head
head_command p4 "$p4_train_cache" "$p4_cache/val.pt" cmd_p4_head
head_command p5 "$p5_cache/train.pt" "$p5_cache/val.pt" cmd_p5_head
cmd_audit=(
  "$python_bin" -B "$audit_runner"
  --p0-dir "$head_root/p0"
  --p4-dir "$head_root/p4"
  --p5-dir "$head_root/p5"
  --p4-pretrain-dir "$p4_pretrain"
  --p5-pretrain-dir "$p5_pretrain"
  --p0-weights "$base_weights"
  --p4-weights "$p4_weights"
  --p5-weights "$p5_pretrain/backbone_best_dev_ap.pth"
  --output-json "$head_root/comparison.json"
  --output-md "$head_root/COMPARISON.md"
)

printf '%s\n' 'Auditing immutable completed P4 inputs:'
require_sha256 "$p4_weights" "$expected_p4_weights_sha256"
require_sha256 "$p4_summary" "$expected_p4_summary_sha256"
require_sha256 "$p4_train_cache" "$expected_p4_train_cache_sha256"
require_sha256 "$p4_train_audit" "$expected_p4_train_audit_sha256"

refuse_existing \
  "$p4_cache/val.pt" \
  "$p4_cache/val.pt.json" \
  "$p5_pretrain" \
  "$p5_cache" \
  "$head_root/p0" \
  "$head_root/p4" \
  "$head_root/p5" \
  "$head_root/comparison.json" \
  "$head_root/COMPARISON.md" \
  "$log_root/resume_preflight_gpu0.log" \
  "$log_root/p4_cache_val.log" \
  "$log_root/p5_pretrain.log" \
  "$log_root/p5_cache_train.log" \
  "$log_root/p5_cache_val.log" \
  "$log_root/p0_head.log" \
  "$log_root/p4_head.log" \
  "$log_root/p5_head.log" \
  "$log_root/audit.log"

if [[ "$execute" -eq 0 ]]; then
  printf '\nDry run only; continuation outputs are absent and no work was started.\n'
  printf 'Protocol: physical GPU0 only; start requires >=%s MiB free; GPU1 hidden.\n' \
    "$minimum_start_free_mib"
  print_command p4-cache-val "${cmd_p4_val_cache[@]}"
  print_command p5-pretrain "${cmd_p5_pretrain[@]}"
  print_command p5-cache-train "${cmd_p5_train_cache[@]}"
  print_command p5-cache-val "${cmd_p5_val_cache[@]}"
  print_command p0-head "${cmd_p0_head[@]}"
  print_command p4-head "${cmd_p4_head[@]}"
  print_command p5-head "${cmd_p5_head[@]}"
  print_command audit "${cmd_audit[@]}"
  exit 0
fi

mkdir -p "$p4_cache" "$head_root" "$log_root"

gpu0_line="$(
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.free \
    --format=csv,noheader,nounits \
    | awk -F, '$1 + 0 == 0 {print}'
)"
if [[ -z "$gpu0_line" ]]; then
  printf 'Could not resolve physical GPU 0 state.\n' >&2
  exit 20
fi
gpu0_uuid="$(awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' <<<"$gpu0_line")"
gpu0_free_mib="$(awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $4); print $4 + 0}' <<<"$gpu0_line")"
if (( gpu0_free_mib < minimum_start_free_mib )); then
  printf 'Refusing to share physical GPU 0: free=%s MiB, required=%s MiB.\n' \
    "$gpu0_free_mib" "$minimum_start_free_mib" >&2
  exit 21
fi

{
  printf 'resume_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf 'script_sha256=%s\n' "$(sha256sum "$0" | awk '{print $1}')"
  printf 'gpu0_state=%s\n' "$gpu0_line"
  printf 'gpu0_external_compute:\n'
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits \
    | awk -F, -v uuid="$gpu0_uuid" '
        {
          observed = $1
          gsub(/^[ \t]+|[ \t]+$/, "", observed)
          if (observed == uuid) print
        }
      '
  printf 'host_memory:\n'
  free -h
  printf 'immutable_inputs:\n'
  sha256sum "$p4_weights" "$p4_summary" "$p4_train_cache" "$p4_train_audit"
} | tee "$log_root/resume_preflight_gpu0.log"

"${cmd_p4_val_cache[@]}" 2>&1 | tee "$log_root/p4_cache_val.log"

"${cmd_p5_pretrain[@]}" 2>&1 | tee "$log_root/p5_pretrain.log"
mkdir -p "$p5_cache"
"${cmd_p5_train_cache[@]}" 2>&1 | tee "$log_root/p5_cache_train.log"
"${cmd_p5_val_cache[@]}" 2>&1 | tee "$log_root/p5_cache_val.log"

"${cmd_p0_head[@]}" 2>&1 | tee "$log_root/p0_head.log"
"${cmd_p4_head[@]}" 2>&1 | tee "$log_root/p4_head.log"
"${cmd_p5_head[@]}" 2>&1 | tee "$log_root/p5_head.log"
"${cmd_audit[@]}" 2>&1 | tee "$log_root/audit.log"
