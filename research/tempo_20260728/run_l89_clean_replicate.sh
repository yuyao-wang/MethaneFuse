#!/usr/bin/env bash
set -euo pipefail

# Full train-only clean-L89 replication. Dry-run is the default.
#
# The formal run is gated by strict local-manifest resolution. Consequently no
# image extraction can begin while the new staging root is incomplete. The
# cross-GPU bit-exact audit establishes hardware interchangeability, but the
# current cross-task boundary locks every formal job to physical GPU 0.

repo_root="/home/yuyao/panopticon"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
output_root="${CLEAN_OUTPUT_ROOT:-/diniuvol/yuyao/methanefuse_research_20260728/l89_clean_replicate_run_v1}"
input_root="$repo_root/research/tempo_20260728/l89_clean_inner_v1"
source_inner_train="$input_root/train.csv"
source_inner_dev="$input_root/dev.csv"
source_inner_readiness="$input_root/READINESS_AUDIT.json"
source_inner_train_sha="8cee38e78f0ff7ebf5e34c02ba07d351cbf93438d9a78e88ad741dc8be3a22eb"
source_inner_dev_sha="8caa5f7a31340c5007ec4457fd5766179a23dc1e97b432322768a8cef0144883"
source_inner_readiness_sha="da54079d348d3d9cf13d3609a61e6315169e75d8b8ee977caa8143901c7c69df"
staging_root="${CLEAN_STAGING_ROOT:-/diniuvol/yuyao/methanefuse_research_20260728/l89_clean_full_local_v1}"
staging_complete_receipt="$staging_root/STAGING_COMPLETE.json"
staging_complete_receipt_sha="3e11d6795c7da28a2c379e51dd993c049e6bb9af37e79b45aae1d9140a68bfec"
manifest_root="$output_root/manifests"
smoke_root="/diniuvol/yuyao/methanefuse_research_20260728/l89_clean_prep_v1/smoke3072"
smoke_manifest_root="$smoke_root/manifests"
smoke_train_cache="$smoke_root/base_train.pt"
smoke_val_cache="$smoke_root/base_val.pt"
base_cache_root="$output_root/base_cache"
sidecar_split_root="$output_root/sidecar_split"
sidecar_root="$output_root/sidecar"
head_root="$output_root/event_balanced_heads_seed20260728"
d1_root="$output_root/tempo_d1_three_seed"
audit_root="$output_root/audits"
log_root="$output_root/logs"
provenance_root="$output_root/provenance"
base_weights="$repo_root/weights/panopticon_vitb14_teacher.pth"
base_train_gpu=0
base_dev_gpu=0
sidecar_p4_gpu=0
sidecar_p5_gpu=0
d1_gpu=0
seeds="20260727,20260728,20260729"
measured_dual_extractor_peak_mib=44671
gpu0_safety_margin_mib=8577
minimum_gpu0_free_mib=53248

resolver="$repo_root/research/tempo_20260728/resolve_l89_clean_manifests.py"
resolver_test="$repo_root/research/tempo_20260728/test_resolve_l89_clean_manifests_cpu.py"
launcher_contract_test="$repo_root/research/tempo_20260728/test_run_l89_clean_replicate_contract_cpu.py"
sidecar_splitter="$repo_root/research/tempo_20260728/prepare_l89_sidecar_inner_split.py"
sidecar_splitter_test="$repo_root/research/tempo_20260728/test_prepare_l89_sidecar_inner_split_cpu.py"
cache_runner="$repo_root/research/pretraining_20260727/l89_ragged_cls_experiment.py"
sidecar_runner="$repo_root/research/pretraining_20260727/rctp_l89_sidecar_fallback.py"
sidecar_runner_test="$repo_root/research/pretraining_20260727/test_rctp_l89_sidecar_fallback_cpu.py"
head_runner="$repo_root/research/pretraining_20260727/rctp_l89_event_balanced_head_followup.py"
head_runner_test="$repo_root/research/pretraining_20260727/test_rctp_l89_event_balanced_head_followup_cpu.py"
d1_runner="$repo_root/research/tempo_20260728/tempo_l89_global.py"
d1_runner_test="$repo_root/research/tempo_20260728/test_tempo_l89_global_cpu.py"
ensemble_runner="$repo_root/research/tempo_20260728/audit_l89_clean_fixed_ensemble.py"
ensemble_test="$repo_root/research/tempo_20260728/test_audit_l89_clean_fixed_ensemble_cpu.py"
promotion_runner="$repo_root/research/tempo_20260728/audit_l89_clean_promotion_chain.py"
promotion_test="$repo_root/research/tempo_20260728/test_audit_l89_clean_promotion_chain_cpu.py"
promotion_root="$audit_root/promotion_chain"
smoke_auditor="$repo_root/research/tempo_20260728/audit_l89_clean_smoke_prerequisites.py"
smoke_receipt="$repo_root/research/tempo_20260728/l89_clean_e2e_smoke3072_v1.json"
smoke_receipt_sha="17d9a77257d3f78856ffa48b052dcc70646f5d8d40637ba882b93cf335bb05bc"
ledger_runner="$repo_root/research/tempo_20260728/audit_l89_clean_model_loads.py"
ledger_test="$repo_root/research/tempo_20260728/test_audit_l89_clean_model_loads_cpu.py"
extraction_benchmark="$repo_root/research/tempo_20260728/l89_clean_extraction_benchmark_v1.json"
extraction_benchmark_sha="1e9b4a16469c5fb4396f54660e74a78844b6bdd08db47b45dd8a569e809aca6b"
protocol="$repo_root/research/tempo_20260728/L89_CLEAN_REPLICATE_PROTOCOL.md"
launcher="$repo_root/research/tempo_20260728/run_l89_clean_replicate.sh"

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
if [[ "$lower_output" =~ (^|[._/-])(test|sealed|holdout|outer)([._/-]|$) ]]; then
  printf 'Refusing held-out-like output path: %s\n' "$output_root" >&2
  exit 3
fi
if [[ "$base_train_gpu" -ne 0 || "$base_dev_gpu" -ne 0 || \
      "$sidecar_p4_gpu" -ne 0 || "$sidecar_p5_gpu" -ne 0 || \
      "$d1_gpu" -ne 0 ]]; then
  printf 'The audited physical GPU mapping changed.\n' >&2
  exit 4
fi
if [[ "$(sha256sum "$extraction_benchmark" | awk '{print $1}')" != \
      "$extraction_benchmark_sha" ]]; then
  printf 'Frozen extraction benchmark SHA mismatch.\n' >&2
  exit 10
fi

verify_exact_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ ! -f "$path" ]]; then
    printf 'Frozen %s is missing: %s\n' "$label" "$path" >&2
    return 11
  fi
  local observed
  observed="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    printf 'Frozen %s SHA mismatch: observed=%s expected=%s path=%s\n' \
      "$label" "$observed" "$expected" "$path" >&2
    return 12
  fi
}

# These three exact payload identities define the only legal source inner
# split. The check occurs before any formal output-root mutation and is
# repeated independently by the resolver, which records it in AUDIT.json.
verify_exact_sha256 \
  "$source_inner_train" "$source_inner_train_sha" "inner train CSV"
verify_exact_sha256 \
  "$source_inner_dev" "$source_inner_dev_sha" "inner dev CSV"
verify_exact_sha256 \
  "$source_inner_readiness" "$source_inner_readiness_sha" \
  "inner READINESS_AUDIT receipt"
verify_exact_sha256 \
  "$staging_complete_receipt" "$staging_complete_receipt_sha" \
  "full-local STAGING_COMPLETE receipt"

if (( minimum_gpu0_free_mib < \
      measured_dual_extractor_peak_mib + gpu0_safety_margin_mib )); then
  printf 'Invalid GPU0 gate: threshold is below measured peak plus margin.\n' >&2
  exit 13
fi

base_cache_command() {
  local csv_path="$1"
  local split="$2"
  local output_cache="$3"
  local maximum_rows="$4"
  local gpu="$5"
  local -n result="$6"
  result=(
    env
    "PYTHONPATH=$repo_root"
    "CUDA_VISIBLE_DEVICES=$gpu"
    "$python_bin" -B "$cache_runner" cache
    --csv "$csv_path"
    --split "$split"
    --output-cache "$output_cache"
    --weights "$base_weights"
    --batch-size 64
    --num-workers 6
    --prefetch-factor 1
    --persistent-workers
    --device cuda:0
    --amp-dtype bfloat16
    --storage-dtype float16
    --local-cache-mode off
    --max-rows "$maximum_rows"
    --row-selection-seed 20260728
    --max-invalid-t0 0
    --max-read-errors 0
    --log-interval 25
  )
}

sidecar_pretrain_command() {
  local arm="$1"
  local destination="$2"
  local gpu="$3"
  local -n result="$4"
  result=(
    timeout 20m
    env
    "PYTHONPATH=$repo_root"
    "CUDA_VISIBLE_DEVICES=$gpu"
    "$python_bin" -B "$sidecar_runner" pretrain
    --arm "$arm"
    --train-csv "$sidecar_split_root/train.csv"
    --dev-csv "$sidecar_split_root/capability.csv"
    --train-cache "$sidecar_split_root/train.pt"
    --dev-cache "$sidecar_split_root/capability.pt"
    --base-weights "$base_weights"
    --output-dir "$destination"
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
    --seed 20260728
    --device cuda:0
    --min-cuda-free-gib 20
    --min-runtime-cuda-free-gib 15
    --max-cuda-allocated-gib 12
    --log-interval 20
  )
}

sidecar_cache_command() {
  local arm="$1"
  local split="$2"
  local checkpoint="$3"
  local -n result="$4"
  result=(
    env
    "PYTHONPATH=$repo_root"
    "$python_bin" -B "$sidecar_runner" build-cache
    --arm "$arm"
    --split "$split"
    --base-cache "$base_cache_root/$split.pt"
    --output-cache "$sidecar_root/cache/$arm/$split.pt"
    --batch-size 512
  )
  if [[ -n "$checkpoint" ]]; then
    result+=(--sidecar-checkpoint "$checkpoint")
  fi
}

base_cache_command "$manifest_root/train.csv" train \
  "$base_cache_root/train.pt" 0 "$base_train_gpu" cmd_base_train
base_cache_command "$manifest_root/dev.csv" val \
  "$base_cache_root/val.pt" 0 "$base_dev_gpu" cmd_base_dev
sidecar_pretrain_command p4_response_scrambled \
  "$sidecar_root/pretrain/p4" "$sidecar_p4_gpu" cmd_p4_pretrain
sidecar_pretrain_command p5_correct_response \
  "$sidecar_root/pretrain/p5" "$sidecar_p5_gpu" cmd_p5_pretrain
sidecar_cache_command p0 train "" cmd_p0_train
sidecar_cache_command p0 val "" cmd_p0_val
sidecar_cache_command p4 train \
  "$sidecar_root/pretrain/p4/sidecar_best_dev_ap.pt" cmd_p4_train
sidecar_cache_command p4 val \
  "$sidecar_root/pretrain/p4/sidecar_best_dev_ap.pt" cmd_p4_val
sidecar_cache_command p5 train \
  "$sidecar_root/pretrain/p5/sidecar_best_dev_ap.pt" cmd_p5_train
sidecar_cache_command p5 val \
  "$sidecar_root/pretrain/p5/sidecar_best_dev_ap.pt" cmd_p5_val

cmd_resolve=(
  "$python_bin" -B "$resolver"
  --input-root "$input_root"
  --output-root "$manifest_root"
  --staging-root "$staging_root"
  --workers 32
)
cmd_smoke_audit=(
  env
  "PYTHONPATH=$repo_root"
  "$python_bin" -B "$smoke_auditor"
  --train-cache "$smoke_train_cache"
  --val-cache "$smoke_val_cache"
  --train-csv "$smoke_manifest_root/train.csv"
  --val-csv "$smoke_manifest_root/dev.csv"
  --base-weights "$base_weights"
  --e2e-receipt "$smoke_receipt"
)
cmd_sidecar_split=(
  env
  "PYTHONPATH=$repo_root"
  "$python_bin" -B "$sidecar_splitter"
  --train-csv "$manifest_root/train.csv"
  --train-cache "$base_cache_root/train.pt"
  --output-root "$sidecar_split_root"
  --minimum-capability-rows 512
)
cmd_heads=(
  env
  "PYTHONPATH=$repo_root"
  "$python_bin" -B "$head_runner"
  --p0-train-cache "$sidecar_root/cache/p0/train.pt"
  --p0-val-cache "$sidecar_root/cache/p0/val.pt"
  --p4-train-cache "$sidecar_root/cache/p4/train.pt"
  --p4-val-cache "$sidecar_root/cache/p4/val.pt"
  --p5-train-cache "$sidecar_root/cache/p5/train.pt"
  --p5-val-cache "$sidecar_root/cache/p5/val.pt"
  --output-dir "$head_root"
  --clean-inner-replicate
  --epochs 3
  --batch-size 256
  --eval-batch-size 512
  --learning-rate 3e-4
  --weight-decay 0.05
  --model-dim 256
  --num-heads 8
  --seed 20260728
  --num-threads 12
  --bootstrap-replicates 2000
)
cmd_d1=(
  env
  "PYTHONPATH=$repo_root"
  "CUDA_VISIBLE_DEVICES=$d1_gpu"
  "$python_bin" -B "$d1_runner" run
  --train-cache "$base_cache_root/train.pt"
  --dev-cache "$base_cache_root/val.pt"
  --base-kind event_balanced_p0
  --event-base-checkpoint \
  "$head_root/p0/checkpoint_best_event_balanced_ap.pt"
  --output-dir "$d1_root"
  --arms p0_base,d1_gated_delta
  --seeds "$seeds"
  --epochs 4
  --patience 1
  --batch-size 512
  --eval-batch-size 1024
  --temporal-dim 192
  --learning-rate 8e-4
  --weight-decay 0.02
  --dropout 0.15
  --device cuda:0
)
cmd_ensemble=(
  env
  "PYTHONPATH=$repo_root"
  "$python_bin" -B "$ensemble_runner"
  --p0-predictions "$d1_root/seed_20260727/p0_base_predictions.csv"
  --p5-predictions \
  "$head_root/p5/validation_best_event_balanced_ap_predictions.csv"
  --d1-template \
  "$d1_root/seed_{seed}/d1_gated_delta_best_event_ap_predictions.csv"
  --seeds "$seeds"
  --replicates 5000
  --bootstrap-seed 2026072808
  --output-dir "$audit_root/fixed_p5_mean_d1"
)
cmd_promotion=(
  env
  "PYTHONPATH=$repo_root"
  "CUDA_VISIBLE_DEVICES="
  "$python_bin" -B "$promotion_runner"
  --p4-sidecar-summary "$sidecar_root/pretrain/p4/summary.json"
  --p4-sidecar-checkpoint \
  "$sidecar_root/pretrain/p4/sidecar_best_dev_ap.pt"
  --p5-sidecar-summary "$sidecar_root/pretrain/p5/summary.json"
  --p5-sidecar-checkpoint \
  "$sidecar_root/pretrain/p5/sidecar_best_dev_ap.pt"
  --head-comparison "$head_root/comparison.json"
  --fixed-ensemble-result "$audit_root/fixed_p5_mean_d1/RESULT.json"
  --p0-predictions "$d1_root/seed_20260727/p0_base_predictions.csv"
  --p0-head-predictions \
  "$head_root/p0/validation_best_event_balanced_ap_predictions.csv"
  --p4-predictions \
  "$head_root/p4/validation_best_event_balanced_ap_predictions.csv"
  --p5-predictions \
  "$head_root/p5/validation_best_event_balanced_ap_predictions.csv"
  --d1-template \
  "$d1_root/seed_{seed}/d1_gated_delta_best_event_ap_predictions.csv"
  --d1-summary-template "$d1_root/seed_{seed}/summary.json"
  --protocol "$protocol"
  --train-manifest "$manifest_root/train.csv"
  --dev-manifest "$manifest_root/dev.csv"
  --split-receipt "$manifest_root/AUDIT.json"
  --model-ledger-final "$output_root/MODEL_LOAD_LEDGER_FINAL.json"
  --base-weights "$base_weights"
  --output-dir "$promotion_root"
)
cmd_ledger_plan=(
  env
  "PYTHONPATH=$repo_root"
  "$python_bin" -B "$ledger_runner"
  --formal-root "$output_root"
  --base-weights "$base_weights"
  --phase plan
)
cmd_ledger_finalize=(
  env
  "PYTHONPATH=$repo_root"
  "$python_bin" -B "$ledger_runner"
  --formal-root "$output_root"
  --base-weights "$base_weights"
  --phase finalize
)

print_command() {
  local label="$1"
  shift
  printf '%-20s' "$label"
  printf ' %q' "$@"
  printf '\n'
}

preflight_resources() {
  local phase="$1"
  gpu_snapshot="$(
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )"
  local gpu_line
  local gpu_free_mib
  gpu_line="$(
    awk -F, '$1 + 0 == 0 {print}' <<<"$gpu_snapshot"
  )"
  if [[ -z "$gpu_line" ]]; then
    printf '%s preflight: required physical GPU 0 is unavailable.\n' \
      "$phase" >&2
    return 6
  fi
  gpu_free_mib="$(
    awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3 + 0}' \
      <<<"$gpu_line"
  )"
  if (( gpu_free_mib < minimum_gpu0_free_mib )); then
    printf '%s preflight refuses GPU0: %s MiB free; at least %s MiB is required (measured dual-extractor peak %s MiB + %s MiB safety margin).\n' \
      "$phase" "$gpu_free_mib" "$minimum_gpu0_free_mib" \
      "$measured_dual_extractor_peak_mib" "$gpu0_safety_margin_mib" >&2
    return 7
  fi
  host_available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( host_available_kib < 41943040 )); then
    printf '%s preflight refuses start: host MemAvailable is below 40 GiB.\n' \
      "$phase" >&2
    return 8
  fi
}

if [[ "$execute" -eq 0 ]]; then
  printf 'Dry run only. No files, images, CPU jobs, or GPU jobs were started.\n'
  printf 'Immutable 3072-row smoke root: %s\n' "$smoke_root"
  printf 'Formal output root: %s\n' "$output_root"
  printf 'Frozen GPU mapping: all formal GPU work -> physical 0; physical 1 is never exposed.\n'
  printf 'Frozen source split SHA: train=%s dev=%s readiness=%s\n' \
    "$source_inner_train_sha" "$source_inner_dev_sha" \
    "$source_inner_readiness_sha"
  printf 'Frozen staging-complete receipt SHA: %s\n' \
    "$staging_complete_receipt_sha"
  printf 'GPU0 launch gate: >=%s MiB free (%s MiB measured dual-extractor peak + %s MiB margin).\n' \
    "$minimum_gpu0_free_mib" "$measured_dual_extractor_peak_mib" \
    "$gpu0_safety_margin_mib"
  print_command validate-smoke3072 "${cmd_smoke_audit[@]}"
  print_command resolve-local "${cmd_resolve[@]}"
  print_command model-ledger-plan "${cmd_ledger_plan[@]}"
  print_command base-train "${cmd_base_train[@]}"
  print_command base-dev "${cmd_base_dev[@]}"
  print_command sidecar-split "${cmd_sidecar_split[@]}"
  print_command p4-pretrain "${cmd_p4_pretrain[@]}"
  print_command p5-pretrain "${cmd_p5_pretrain[@]}"
  print_command p0-train-cache "${cmd_p0_train[@]}"
  print_command p0-val-cache "${cmd_p0_val[@]}"
  print_command p4-train-cache "${cmd_p4_train[@]}"
  print_command p4-val-cache "${cmd_p4_val[@]}"
  print_command p5-train-cache "${cmd_p5_train[@]}"
  print_command p5-val-cache "${cmd_p5_val[@]}"
  print_command matched-heads "${cmd_heads[@]}"
  print_command d1-three-seed "${cmd_d1[@]}"
  print_command primary-ensemble "${cmd_ensemble[@]}"
  print_command model-ledger-final "${cmd_ledger_finalize[@]}"
  print_command promotion-chain "${cmd_promotion[@]}"
  exit 0
fi

if [[ -e "$output_root" ]]; then
  printf 'Refusing any previous formal lineage root: %s\n' "$output_root" >&2
  exit 5
fi

# This read-only gate runs before resolver, mkdir, ledger, or any other formal
# output-root mutation. A busy resource therefore leaves no poisoned lineage.
preflight_resources "initial"
initial_gpu_snapshot="$gpu_snapshot"
initial_host_available_kib="$host_available_kib"

# CPU contract tests occur before any staging path is resolved or image read.
"$python_bin" -m unittest \
  research.tempo_20260728.test_resolve_l89_clean_manifests_cpu \
  research.tempo_20260728.test_run_l89_clean_replicate_contract_cpu \
  research.tempo_20260728.test_prepare_l89_sidecar_inner_split_cpu \
  research.tempo_20260728.test_audit_l89_clean_fixed_ensemble_cpu \
  research.tempo_20260728.test_audit_l89_clean_promotion_chain_cpu \
  research.tempo_20260728.test_audit_l89_clean_model_loads_cpu \
  research.pretraining_20260727.test_rctp_l89_sidecar_fallback_cpu \
  research.pretraining_20260727.test_rctp_l89_event_balanced_head_followup_cpu \
  research.tempo_20260728.test_tempo_l89_global_cpu -v

if [[ "$(sha256sum "$smoke_receipt" | awk '{print $1}')" != \
      "$smoke_receipt_sha" ]]; then
  printf 'Frozen same-GPU end-to-end smoke receipt SHA mismatch.\n' >&2
  exit 9
fi
smoke_audit_json="$("${cmd_smoke_audit[@]}")"

# This is the hard formal staging-complete gate. Any missing full-cohort file
# exits before creating either formal manifest.
"${cmd_resolve[@]}"

mkdir -p \
  "$base_cache_root" \
  "$sidecar_root/cache/p0" \
  "$sidecar_root/cache/p4" \
  "$sidecar_root/cache/p5" \
  "$audit_root" \
  "$log_root" \
  "$provenance_root"

source_files=(
  "$resolver"
  "$resolver_test"
  "$launcher_contract_test"
  "$sidecar_splitter"
  "$sidecar_splitter_test"
  "$cache_runner"
  "$sidecar_runner"
  "$sidecar_runner_test"
  "$head_runner"
  "$head_runner_test"
  "$d1_runner"
  "$d1_runner_test"
  "$ensemble_runner"
  "$ensemble_test"
  "$promotion_runner"
  "$promotion_test"
  "$smoke_auditor"
  "$smoke_receipt"
  "$ledger_runner"
  "$ledger_test"
  "$extraction_benchmark"
  "$launcher"
  "$protocol"
)
sha256sum "${source_files[@]}" > "$provenance_root/SOURCE_SHA256SUMS.txt"
sha256sum \
  "$source_inner_train" \
  "$source_inner_dev" \
  "$source_inner_readiness" \
  "$staging_complete_receipt" \
  "$manifest_root/train.csv" \
  "$manifest_root/dev.csv" \
  "$smoke_manifest_root/train.csv" \
  "$smoke_manifest_root/dev.csv" \
  "$smoke_train_cache" \
  "$smoke_train_cache.json" \
  "$smoke_val_cache" \
  "$smoke_val_cache.json" \
  "$smoke_receipt" \
  "$extraction_benchmark" \
  "$base_weights" \
  > "$provenance_root/INPUT_SHA256SUMS.txt"
printf '%s\n' "$smoke_audit_json" \
  > "$provenance_root/SMOKE3072_PREREQUISITE_AUDIT.json"

# Lock every permitted formal model load before the first formal PTH is read.
"${cmd_ledger_plan[@]}" >"$log_root/model_load_ledger_plan.log" 2>&1

sampler_pid=""
start_sampler() {
  local output="$1"
  (
    while true; do
      date -u +'%Y-%m-%dT%H:%M:%SZ'
      nvidia-smi \
        --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 5
    done
  ) >>"$output" 2>&1 &
  sampler_pid=$!
}
stop_sampler() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=""
  fi
}
trap stop_sampler EXIT INT TERM

run_pair() {
  local first_label="$1"
  local first_log="$2"
  local first_array_name="$3"
  local second_label="$4"
  local second_log="$5"
  local second_array_name="$6"
  local -n first_command="$first_array_name"
  local -n second_command="$second_array_name"
  printf 'Starting concurrent %s and %s\n' "$first_label" "$second_label"
  "${first_command[@]}" >"$first_log" 2>&1 &
  local first_pid=$!
  "${second_command[@]}" >"$second_log" 2>&1 &
  local second_pid=$!
  local completed_pid=""
  local first_status=0
  set +e
  wait -n -p completed_pid "$first_pid" "$second_pid"
  first_status=$?
  set -e
  if [[ "$first_status" -ne 0 ]]; then
    kill "$first_pid" "$second_pid" 2>/dev/null || true
    wait "$first_pid" "$second_pid" 2>/dev/null || true
    printf 'Concurrent pair failed first at PID %s; see %s and %s\n' \
      "$completed_pid" "$first_log" "$second_log" >&2
    return "$first_status"
  fi
  local remaining_pid="$first_pid"
  if [[ "$completed_pid" == "$first_pid" ]]; then
    remaining_pid="$second_pid"
  fi
  set +e
  wait "$remaining_pid"
  local remaining_status=$?
  set -e
  if [[ "$remaining_status" -ne 0 ]]; then
    printf 'Concurrent pair second process failed; see %s and %s\n' \
      "$first_log" "$second_log" >&2
    return "$remaining_status"
  fi
}

# Repeat the full GPU0/host gate immediately before the concurrent base pair,
# after all CPU-only contracts and lineage receipts. This is deliberately
# adjacent to start_sampler/run_pair to close the resource-claim race window.
preflight_resources "base-pair"
base_pair_gpu_snapshot="$gpu_snapshot"
base_pair_host_available_kib="$host_available_kib"
{
  printf 'started_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf 'base_train_physical_gpu=0\n'
  printf 'base_dev_physical_gpu=0\n'
  printf 'sidecar_p4_physical_gpu=0\n'
  printf 'sidecar_p5_physical_gpu=0\n'
  printf 'd1_physical_gpu=0\n'
  printf 'minimum_free_mib_per_gpu=%s\n' "$minimum_gpu0_free_mib"
  printf 'measured_dual_extractor_peak_mib=%s\n' \
    "$measured_dual_extractor_peak_mib"
  printf 'gpu0_safety_margin_mib=%s\n' "$gpu0_safety_margin_mib"
  printf 'initial_gpu_preflight=%s\n' "$initial_gpu_snapshot"
  printf 'initial_host_mem_available_kib=%s\n' \
    "$initial_host_available_kib"
  printf 'base_pair_gpu_preflight=%s\n' "$base_pair_gpu_snapshot"
  printf 'base_pair_host_mem_available_kib=%s\n' \
    "$base_pair_host_available_kib"
  printf 'base_extraction_batch_per_split=64\n'
  printf 'base_extraction_workers_per_split=6\n'
  printf 'base_extraction_prefetch_factor=1\n'
  printf 'measured_simultaneous_512_rows_b64_seconds=57.4\n'
  printf 'measured_simultaneous_512_rows_b128_seconds=66.3\n'
  printf 'measured_simultaneous_512_rows_b256_seconds=122.6\n'
  printf 'batch64_same_setting_repeat_bit_exact=true\n'
  printf 'batch64_cross_gpu_bit_exact=true\n'
  printf 'extraction_benchmark_sha256=%s\n' "$extraction_benchmark_sha"
} > "$log_root/preflight.log"

start_sampler "$log_root/gpu_base_cache.csv"
run_pair base-train "$log_root/base_train.log" cmd_base_train \
  base-dev "$log_root/base_dev.log" cmd_base_dev
stop_sampler

"${cmd_sidecar_split[@]}" >"$log_root/sidecar_inner_split.log" 2>&1

start_sampler "$log_root/gpu_sidecar_pretrain.csv"
run_pair p4-pretrain "$log_root/p4_pretrain.log" cmd_p4_pretrain \
  p5-pretrain "$log_root/p5_pretrain.log" cmd_p5_pretrain
stop_sampler

run_pair p0-train-cache "$log_root/p0_train_cache.log" cmd_p0_train \
  p0-val-cache "$log_root/p0_val_cache.log" cmd_p0_val
run_pair p4-train-cache "$log_root/p4_train_cache.log" cmd_p4_train \
  p4-val-cache "$log_root/p4_val_cache.log" cmd_p4_val
run_pair p5-train-cache "$log_root/p5_train_cache.log" cmd_p5_train \
  p5-val-cache "$log_root/p5_val_cache.log" cmd_p5_val

"${cmd_heads[@]}" >"$log_root/event_balanced_heads.log" 2>&1
start_sampler "$log_root/gpu_d1.csv"
"${cmd_d1[@]}" >"$log_root/d1_three_seed.log" 2>&1
stop_sampler
"${cmd_ensemble[@]}" >"$log_root/fixed_ensemble_5000_bootstrap.log" 2>&1
"${cmd_ledger_finalize[@]}" >"$log_root/model_load_ledger_finalize.log" 2>&1
"${cmd_promotion[@]}" >"$log_root/promotion_chain.log" 2>&1

sha256sum -c "$provenance_root/SOURCE_SHA256SUMS.txt"
sha256sum -c "$provenance_root/INPUT_SHA256SUMS.txt"
artifact_files=(
  "$provenance_root/SOURCE_SHA256SUMS.txt" \
  "$provenance_root/INPUT_SHA256SUMS.txt" \
  "$provenance_root/SMOKE3072_PREREQUISITE_AUDIT.json" \
  "$manifest_root/AUDIT.json" \
  "$base_cache_root/train.pt" \
  "$base_cache_root/val.pt" \
  "$sidecar_split_root/AUDIT.json" \
  "$sidecar_split_root/train.pt" \
  "$sidecar_split_root/capability.pt" \
  "$sidecar_root/pretrain/p4/summary.json" \
  "$sidecar_root/pretrain/p4/sidecar_best_dev_ap.pt" \
  "$sidecar_root/pretrain/p5/summary.json" \
  "$sidecar_root/pretrain/p5/sidecar_best_dev_ap.pt" \
  "$sidecar_root/cache/p0/train.pt" \
  "$sidecar_root/cache/p0/val.pt" \
  "$sidecar_root/cache/p4/train.pt" \
  "$sidecar_root/cache/p4/val.pt" \
  "$sidecar_root/cache/p5/train.pt" \
  "$sidecar_root/cache/p5/val.pt" \
  "$head_root/comparison.json" \
  "$head_root/p0/validation_best_event_balanced_ap_predictions.csv" \
  "$head_root/p4/validation_best_event_balanced_ap_predictions.csv" \
  "$head_root/p5/validation_best_event_balanced_ap_predictions.csv" \
  "$d1_root/aggregate.json" \
  "$d1_root/seed_20260727/p0_base_predictions.csv" \
  "$d1_root/seed_20260727/summary.json" \
  "$d1_root/seed_20260727/d1_gated_delta_metrics_history.json" \
  "$d1_root/seed_20260728/p0_base_predictions.csv" \
  "$d1_root/seed_20260728/summary.json" \
  "$d1_root/seed_20260728/d1_gated_delta_metrics_history.json" \
  "$d1_root/seed_20260729/p0_base_predictions.csv" \
  "$d1_root/seed_20260729/summary.json" \
  "$d1_root/seed_20260729/d1_gated_delta_metrics_history.json" \
  "$audit_root/fixed_p5_mean_d1/RESULT.json" \
  "$audit_root/fixed_p5_mean_d1/fixed_p5_mean_d1_predictions.csv" \
  "$promotion_root/RESULT.json" \
  "$promotion_root/PROMOTION_DECISION.json" \
  "$promotion_root/RESULT.md" \
  "$output_root/MODEL_LOAD_LEDGER.json" \
  "$output_root/MODEL_LOAD_LEDGER_FINAL.json"
)
if [[ -f "$promotion_root/fixed_p0_mean_d1_predictions.csv" ]]; then
  artifact_files+=("$promotion_root/fixed_p0_mean_d1_predictions.csv")
fi
sha256sum "${artifact_files[@]}" > "$output_root/ARTIFACT_SHA256SUMS.txt"

{
  printf '{\n'
  printf '  "status": "complete",\n'
  printf '  "completed_utc": "%s",\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf '  "physical_gpu_mapping": {"base_train": 0, "base_dev": 0, "sidecar_p4": 0, "sidecar_p5": 0, "d1": 0},\n'
  printf '  "d1_seeds": [20260727, 20260728, 20260729],\n'
  printf '  "fixed_candidate_weight": 0.5,\n'
  printf '  "staging_complete_receipt_sha256": "%s",\n' \
    "$staging_complete_receipt_sha"
  printf '  "promotion_chain_result_sha256": "%s",\n' \
    "$(sha256sum "$promotion_root/RESULT.json" | awk '{print $1}')"
  printf '  "test_or_sealed_or_holdout_or_outer_read": false\n'
  printf '}\n'
} > "$output_root/RUN_COMPLETE.json"
printf 'Clean L89 replication complete: %s\n' "$output_root"
