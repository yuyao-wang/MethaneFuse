#!/usr/bin/env bash
set -euo pipefail

# Development-only sweep for the cached legacy-360 two-axis head.
#
# Safety properties:
#   * default invocation is a dry-run; --run is required to start work;
#   * only train_core and dev feature caches are accepted;
#   * no test/sealed-test argument exists in this launcher;
#   * every run has its own output directory, stdout/stderr log, trainer
#     run_status.json, and launcher_status.json;
#   * existing run artifacts are never overwritten.

usage() {
  cat <<'EOF'
Usage:
  ./run_two_axis_head_sweep.sh             # print the 10-run plan only
  ./run_two_axis_head_sweep.sh --run       # validate caches and launch

Optional environment overrides:
  TRAIN_CACHE       merged train_core feature cache
  DEV_CACHE         development feature cache
  HEADS_ROOT        destination directory for independent runs
  GPU_IDS           comma-separated physical GPU indices (default: 0,1)
  MAX_HEADS_PER_GPU concurrent head processes per GPU (default: 2)
  PYTHON_BIN        Python interpreter
  RUNNER            two-axis runner path
  FORMAL_ROOT       formal experiment root
  REPO_ROOT         panopticon repository
  BATCH_SIZE        training batch size (default: 512)
  EVAL_BATCH_SIZE   development evaluation batch size (default: 2048)

This launcher never reads or accepts a test cache. Sealed evaluation must be
performed later, once, with evaluate-locked and a selected selection_lock.json.
EOF
}

mode=plan
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --run)
      mode=run
      ;;
    --plan)
      mode=plan
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
fi

repo_root=${REPO_ROOT:-/home/yuyao/panopticon}
python_bin=${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}
formal_root=${FORMAL_ROOT:-/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5}
train_cache=${TRAIN_CACHE:-${formal_root}/features/train_core_universal_s2hybrid.pt}
dev_cache=${DEV_CACHE:-${formal_root}/features/dev_universal_s2hybrid.pt}
heads_root=${HEADS_ROOT:-${formal_root}/heads}
runner=${RUNNER:-${repo_root}/research/pretraining_20260727/query360_two_axis_full_legacy.py}
gpu_ids=${GPU_IDS:-0,1}
max_heads_per_gpu=${MAX_HEADS_PER_GPU:-2}
batch_size=${BATCH_SIZE:-512}
eval_batch_size=${EVAL_BATCH_SIZE:-2048}

epochs=3
seed=42
dropout=0.1
selection_metric=best_binary_f1

if [[ ! "$max_heads_per_gpu" =~ ^[1-3]$ ]]; then
  echo "MAX_HEADS_PER_GPU must be 1, 2, or 3; got: $max_heads_per_gpu" >&2
  exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer; got: $batch_size" >&2
  exit 2
fi
if [[ ! "$eval_batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_BATCH_SIZE must be a positive integer; got: $eval_batch_size" >&2
  exit 2
fi

IFS=',' read -r -a gpu_array <<< "$gpu_ids"
if [[ ${#gpu_array[@]} -eq 0 ]]; then
  echo "GPU_IDS produced an empty GPU list" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpu_array[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS entries must be non-negative integers; got: $gpu" >&2
    exit 2
  fi
  if [[ -n ${seen_gpus[$gpu]+present} ]]; then
    echo "GPU_IDS must not contain duplicates; got: $gpu_ids" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

guard_development_path() {
  local role=$1
  local value=$2
  local lower=${value,,}
  if [[ "$lower" == *sealed* ]] || [[ "$lower" =~ (^|[/_.-])test([/_.-]|$) ]]; then
    echo "Refusing $role path that looks like test/sealed data: $value" >&2
    exit 2
  fi
}
guard_development_path TRAIN_CACHE "$train_cache"
guard_development_path DEV_CACHE "$dev_cache"

# Five matched arms/base choices crossed with two learning rates.
run_names=(
  hybrid_current_only_lr1e4
  hybrid_current_only_lr3e4
  hybrid_two_axis_lr1e4
  hybrid_two_axis_lr3e4
  hybrid_scale_aware_lr1e4
  hybrid_scale_aware_lr3e4
  universal_current_only_lr1e4
  universal_current_only_lr3e4
  universal_scale_aware_lr1e4
  universal_scale_aware_lr3e4
)
base_modes=(
  hybrid hybrid
  hybrid hybrid
  hybrid hybrid
  universal universal
  universal universal
)
arms=(
  current_only current_only
  two_axis_query two_axis_query
  scale_aware_two_axis_query scale_aware_two_axis_query
  current_only current_only
  scale_aware_two_axis_query scale_aware_two_axis_query
)
learning_rates=(
  1e-4 3e-4
  1e-4 3e-4
  1e-4 3e-4
  1e-4 3e-4
  1e-4 3e-4
)

print_plan() {
  printf 'Protocol: dev-only, epochs=%s seed=%s dropout=%s selection=%s\n' \
    "$epochs" "$seed" "$dropout" "$selection_metric"
  printf 'Train cache: %s\nDev cache:   %s\nHeads root:  %s\n' \
    "$train_cache" "$dev_cache" "$heads_root"
  printf 'Concurrency: %s head(s) per GPU on GPU_IDS=%s\n' \
    "$max_heads_per_gpu" "$gpu_ids"
  printf '\n%-38s %-10s %-30s %-8s\n' RUN BASE ARM LR
  for index in "${!run_names[@]}"; do
    printf '%-38s %-10s %-30s %-8s\n' \
      "${run_names[$index]}" \
      "${base_modes[$index]}" \
      "${arms[$index]}" \
      "${learning_rates[$index]}"
  done
}

if [[ "$mode" == plan ]]; then
  print_plan
  printf '\nDry-run only. Reinvoke with --run after the merged train/dev caches exist.\n'
  exit 0
fi

for required in "$python_bin" "$runner" "$train_cache" "$dev_cache"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

# Sidecar-plus-SHA preflight avoids materializing multi-gigabyte tensors. Both
# extractor and strict-merge commands produce an adjacent .audit.json that
# records the split, cache SHA, feature shape, and encoder provenance.
"$python_bin" - "$train_cache" "$dev_cache" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value):
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__type__": type(value).__name__, "__repr__": repr(value)}


def fingerprint(value):
    canonical = json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def exact_path(value):
    return os.path.abspath(os.path.expanduser(str(value)))


def read_audit(cache_path, role):
    cache = Path(cache_path).expanduser().absolute()
    audit_path = Path(str(cache) + ".audit.json")
    if not audit_path.is_file():
        raise SystemExit(f"{role} cache audit is missing: {audit_path}")
    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    schema = audit.get("schema_version")
    if schema == "legacy360-merged-feature-cache-audit-v1":
        output = audit.get("output", {})
        compatibility = audit.get("compatibility", {})
        tensor_fields = compatibility.get("tensor_fields", {})
        feature = tensor_fields.get("features", {})
        metadata = {
            "split": audit.get("split"),
            "path": output.get("path"),
            "sha256": output.get("sha256"),
            "rows": output.get("rows"),
            "shape": feature.get("shape"),
            "encoder_fingerprint": compatibility.get(
                "encoder_fingerprint"
            ),
            "sealed_authorized": False,
        }
    elif schema == "query360-two-axis-feature-cache-v1":
        encoder = audit.get("encoder")
        metadata = {
            "split": audit.get("split"),
            "path": audit.get("cache"),
            "sha256": audit.get("cache_sha256"),
            "rows": audit.get("rows"),
            "shape": audit.get("feature_shape"),
            "encoder_fingerprint": (
                fingerprint(encoder) if isinstance(encoder, dict) else None
            ),
            "sealed_authorized": bool(
                audit.get("extraction", {}).get(
                    "sealed_test_authorized", False
                )
            ),
        }
    else:
        raise SystemExit(
            f"{role} cache audit has unsupported schema={schema!r}: "
            f"{audit_path}"
        )
    if exact_path(metadata["path"]) != str(cache):
        raise SystemExit(
            f"{role} cache audit path mismatch: {metadata['path']!r} "
            f"vs {cache}"
        )
    if not isinstance(metadata["sha256"], str):
        raise SystemExit(f"{role} cache audit has no SHA-256")
    observed_sha = sha256_file(cache)
    if observed_sha != metadata["sha256"]:
        raise SystemExit(
            f"{role} cache SHA mismatch: observed={observed_sha}, "
            f"audit={metadata['sha256']}"
        )
    shape = metadata["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or int(shape[0]) != int(metadata["rows"])
    ):
        raise SystemExit(f"{role} cache audit has invalid feature shape")
    if not isinstance(metadata["encoder_fingerprint"], str):
        raise SystemExit(f"{role} cache audit has no encoder fingerprint")
    return metadata


expected = (
    ("train", {"train", "train_core"}),
    ("dev", {"dev", "evaluation"}),
)
metadata = []
for path, (role, allowed) in zip(sys.argv[1:], expected):
    item = read_audit(path, role)
    if item["split"] not in allowed:
        raise SystemExit(
            f"{role} cache split={item['split']!r}, "
            f"expected {sorted(allowed)}"
        )
    if item["sealed_authorized"]:
        raise SystemExit(f"{role} cache audit records held-out authorization")
    metadata.append(item)
if (
    metadata[0]["encoder_fingerprint"]
    != metadata[1]["encoder_fingerprint"]
):
    raise SystemExit("train/dev encoder provenance differs")
if tuple(metadata[0]["shape"][1:]) != tuple(metadata[1]["shape"][1:]):
    raise SystemExit("train/dev feature shapes differ")
print(
    "cache SHA/audit preflight passed:",
    metadata[0]["split"],
    int(metadata[0]["rows"]),
    "rows;",
    metadata[1]["split"],
    int(metadata[1]["rows"]),
    "rows",
)
PY

mkdir -p "$heads_root"
for run_name in "${run_names[@]}"; do
  run_dir=${heads_root}/${run_name}
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse existing run directory: $run_dir" >&2
    exit 2
  fi
done

write_launcher_status() {
  local destination=$1
  local state=$2
  local exit_code=$3
  local run_name=$4
  local gpu=$5
  local base_mode=$6
  local arm=$7
  local learning_rate=$8
  "$python_bin" - \
    "$destination" "$state" "$exit_code" "$run_name" "$gpu" \
    "$base_mode" "$arm" "$learning_rate" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    destination,
    state,
    exit_code,
    run_name,
    gpu,
    base_mode,
    arm,
    learning_rate,
) = sys.argv[1:]
payload = {
    "schema_version": "query360-two-axis-head-launch-status-v1",
    "status": state,
    "utc": datetime.now(timezone.utc).isoformat(),
    "exit_code": int(exit_code),
    "run": run_name,
    "gpu": int(gpu),
    "base_mode": base_mode,
    "arm": arm,
    "learning_rate": float(learning_rate),
    "dev_only": True,
    "sealed_test_read": False,
    "sealed_test_evaluations": 0,
}
temporary = destination + f".tmp.{os.getpid()}"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, destination)
PY
}

run_one() {
  local index=$1
  local gpu=$2
  local run_name=${run_names[$index]}
  local base_mode=${base_modes[$index]}
  local arm=${arms[$index]}
  local learning_rate=${learning_rates[$index]}
  local run_dir=${heads_root}/${run_name}
  local log_path=${run_dir}/stdout_stderr.log
  local launch_status=${run_dir}/launcher_status.json

  mkdir -p "$run_dir"
  write_launcher_status \
    "$launch_status" running 0 "$run_name" "$gpu" \
    "$base_mode" "$arm" "$learning_rate"

  {
    printf '[launcher] run=%s gpu=%s base=%s arm=%s lr=%s\n' \
      "$run_name" "$gpu" "$base_mode" "$arm" "$learning_rate"
    printf '[launcher] train_cache=%s\n[launcher] dev_cache=%s\n' \
      "$train_cache" "$dev_cache"
    printf '[launcher] command:'
    printf ' %q' \
      "$python_bin" "$runner" train \
      --train-cache "$train_cache" \
      --dev-cache "$dev_cache" \
      --output-dir "$run_dir" \
      --base-mode "$base_mode" \
      --arm "$arm" \
      --epochs "$epochs" \
      --seed "$seed" \
      --batch-size "$batch_size" \
      --eval-batch-size "$eval_batch_size" \
      --learning-rate "$learning_rate" \
      --dropout "$dropout" \
      --selection-metric "$selection_metric" \
      --device "cuda:${gpu}"
    printf '\n'

    set +e
    "$python_bin" "$runner" train \
      --train-cache "$train_cache" \
      --dev-cache "$dev_cache" \
      --output-dir "$run_dir" \
      --base-mode "$base_mode" \
      --arm "$arm" \
      --epochs "$epochs" \
      --seed "$seed" \
      --batch-size "$batch_size" \
      --eval-batch-size "$eval_batch_size" \
      --learning-rate "$learning_rate" \
      --dropout "$dropout" \
      --selection-metric "$selection_metric" \
      --device "cuda:${gpu}"
    local exit_code=$?
    set -e

    local final_state=failed
    if [[ $exit_code -eq 0 ]] && \
       "$python_bin" - \
         "$run_dir" "$base_mode" "$arm" "$learning_rate" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
base_mode = sys.argv[2]
arm = sys.argv[3]
learning_rate = float(sys.argv[4])
required = (
    "run_status.json",
    "summary.json",
    "selection_lock.json",
    "checkpoint_best.pth",
)
missing = [name for name in required if not (run_dir / name).is_file()]
if missing:
    raise SystemExit(f"runner artifacts missing: {missing}")
with (run_dir / "run_status.json").open(encoding="utf-8") as stream:
    status = json.load(stream)
with (run_dir / "summary.json").open(encoding="utf-8") as stream:
    summary = json.load(stream)
with (run_dir / "selection_lock.json").open(encoding="utf-8") as stream:
    lock = json.load(stream)
if status.get("status") != "complete":
    raise SystemExit("runner status is not complete")
for payload, role in ((status, "status"), (summary, "summary")):
    if bool(payload.get("sealed_test_read", False)):
        raise SystemExit(f"{role} records a held-out read")
    if int(payload.get("sealed_test_evaluations", 0)) != 0:
        raise SystemExit(f"{role} records a held-out evaluation")
if lock.get("test_cache_read_before_lock") is not False:
    raise SystemExit("selection lock lacks the no-held-out-read assertion")
if lock.get("base_mode") != base_mode:
    raise SystemExit("selection lock base mode mismatch")
if lock.get("arm") != arm:
    raise SystemExit("selection lock arm mismatch")
model = summary.get("model", {})
training = summary.get("training", {})
if model.get("base_mode") != base_mode:
    raise SystemExit("summary base mode mismatch")
if model.get("arm") != arm:
    raise SystemExit("summary arm mismatch")
if int(training.get("epochs", -1)) != 3:
    raise SystemExit("summary epoch budget mismatch")
if not math.isclose(
    float(training.get("learning_rate", -1.0)),
    learning_rate,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit("summary learning rate mismatch")
PY
    then
      final_state=complete
    elif [[ $exit_code -eq 0 ]]; then
      exit_code=3
    fi
    write_launcher_status \
      "$launch_status" "$final_state" "$exit_code" "$run_name" "$gpu" \
      "$base_mode" "$arm" "$learning_rate"
    printf '[launcher] final_state=%s exit_code=%s\n' \
      "$final_state" "$exit_code"
    return "$exit_code"
  } >"$log_path" 2>&1
}

print_plan
printf '\nLaunching development-only sweep.\n'

# Deterministic wave scheduling. Each GPU receives MAX_HEADS_PER_GPU jobs in
# parallel; the next wave starts after the current wave has fully completed.
slots=()
for gpu in "${gpu_array[@]}"; do
  for ((slot = 0; slot < max_heads_per_gpu; slot += 1)); do
    slots+=("$gpu")
  done
done

overall_exit=0
for ((wave_start = 0; wave_start < ${#run_names[@]}; wave_start += ${#slots[@]})); do
  wave_pids=()
  wave_names=()
  for slot_index in "${!slots[@]}"; do
    run_index=$((wave_start + slot_index))
    if ((run_index >= ${#run_names[@]})); then
      break
    fi
    run_one "$run_index" "${slots[$slot_index]}" &
    wave_pids+=("$!")
    wave_names+=("${run_names[$run_index]}")
  done
  for pid_index in "${!wave_pids[@]}"; do
    if ! wait "${wave_pids[$pid_index]}"; then
      echo "Run failed: ${wave_names[$pid_index]}" >&2
      overall_exit=1
    fi
  done
done

if [[ $overall_exit -ne 0 ]]; then
  echo "Sweep finished with one or more failures; inspect independent logs." >&2
  exit "$overall_exit"
fi
echo "All development-only head runs completed."
