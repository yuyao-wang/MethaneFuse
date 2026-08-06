#!/usr/bin/env bash
set -euo pipefail

# Four-candidate promotion from the completed pilot to the complete
# train_core cache. Development only: this launcher has no held-out input.
# Every runner retains epoch 0 as an eligible exact-PTH fallback.

usage() {
  cat <<'EOF'
Usage:
  ./run_full_cache_dev_promotion.sh           # dry-run plan (default)
  ./run_full_cache_dev_promotion.sh --plan    # dry-run plan
  ./run_full_cache_dev_promotion.sh --run     # validate and launch

Exactly four candidates:
  1. universal scale-aware gated delta, lr=1e-4, cap=1.5, bottleneck=32
  2. universal scale-aware gated delta, lr=3e-4, cap=1.5, bottleneck=32
  3. compact universal scale-aware two-axis, d=64, depth=1, lr=1e-4
  4. compact universal two-axis, d=64, depth=1, lr=3e-5

All candidates use dropout=0 for compact axial heads, at most three epochs,
development-only selection, and an eligible epoch-0 checkpoint base.

Optional environment:
  FORMAL_ROOT          experiment root
  TRAIN_CACHE          canonical complete train_core cache
  DEV_CACHE            canonical development cache
  HEADS_ROOT           new, non-existing promotion output root
  GPU_IDS              comma-separated physical GPU IDs (default: 0,1)
  MAX_HEADS_PER_GPU    1 or 2 concurrent heads per GPU (default: 1)
  EPOCHS               1, 2, or 3 only (default: 3)
  BATCH_SIZE           positive integer (default: 1024)
  EVAL_BATCH_SIZE      positive integer (default: 4096)
  PYTHON_BIN           Python interpreter
  GATED_RUNNER         gated-delta runner
  TWO_AXIS_RUNNER      compact two-axis runner
  REPO_ROOT            panopticon repository

The run mode refuses noncanonical/pilot cache names and any existing
HEADS_ROOT. There is no resume or overwrite mode.
EOF
}

mode=plan
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --plan)
      mode=plan
      ;;
    --run)
      mode=run
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
heads_root=${HEADS_ROOT:-${formal_root}/heads/full_cache_dev_promotion_v1}
gated_runner=${GATED_RUNNER:-${repo_root}/research/pretraining_20260727/query360_gated_delta_runner.py}
two_axis_runner=${TWO_AXIS_RUNNER:-${repo_root}/research/pretraining_20260727/query360_two_axis_full_legacy.py}
gpu_ids=${GPU_IDS:-0,1}
max_heads_per_gpu=${MAX_HEADS_PER_GPU:-1}
epochs=${EPOCHS:-3}
batch_size=${BATCH_SIZE:-1024}
eval_batch_size=${EVAL_BATCH_SIZE:-4096}

seed=42
selection_metric=best_binary_f1

if [[ ! "$epochs" =~ ^[123]$ ]]; then
  echo "EPOCHS must be 1, 2, or 3; got: $epochs" >&2
  exit 2
fi
if [[ ! "$max_heads_per_gpu" =~ ^[12]$ ]]; then
  echo "MAX_HEADS_PER_GPU must be 1 or 2; got: $max_heads_per_gpu" >&2
  exit 2
fi
for pair in \
  "BATCH_SIZE:$batch_size" \
  "EVAL_BATCH_SIZE:$eval_batch_size"; do
  name=${pair%%:*}
  value=${pair#*:}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; got: $value" >&2
    exit 2
  fi
done

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
  if [[ "$lower" == *sealed* ]] ||
     [[ "$lower" =~ (^|[/_.-])test([/_.-]|$) ]]; then
    echo "Refusing $role path that looks held out: $value" >&2
    exit 2
  fi
}
guard_full_cache_path() {
  local role=$1
  local value=$2
  local expected_basename=$3
  local lower=${value,,}
  if [[ "$lower" =~ (^|[/_.-])(pilot|subset|screen|chunk)([/_.-]|$) ]]; then
    echo "Refusing non-full $role cache path: $value" >&2
    exit 2
  fi
  if [[ $(basename "$value") != "$expected_basename" ]]; then
    echo "$role cache must be named $expected_basename; got: $value" >&2
    exit 2
  fi
}
guard_development_path TRAIN_CACHE "$train_cache"
guard_development_path DEV_CACHE "$dev_cache"
guard_development_path HEADS_ROOT "$heads_root"
guard_full_cache_path \
  TRAIN_CACHE "$train_cache" train_core_universal_s2hybrid.pt
guard_full_cache_path \
  DEV_CACHE "$dev_cache" dev_universal_s2hybrid.pt

run_names=(
  promote_gdelta_universal_lr1e4_cap1p5
  promote_gdelta_universal_lr3e4_cap1p5
  promote_compact_scale_aware_d64_lr1e4
  promote_compact_two_axis_d64_lr3e5
)
families=(gated_delta gated_delta compact_axial compact_axial)
arms=(
  scale_aware_gated_delta
  scale_aware_gated_delta
  scale_aware_two_axis_query
  two_axis_query
)
learning_rates=(1e-4 3e-4 1e-4 3e-5)
residual_caps=(1.5 1.5 none none)

if [[ ${#run_names[@]} -ne 4 ||
      ${#families[@]} -ne ${#run_names[@]} ||
      ${#arms[@]} -ne ${#run_names[@]} ||
      ${#learning_rates[@]} -ne ${#run_names[@]} ||
      ${#residual_caps[@]} -ne ${#run_names[@]} ]]; then
  echo "Internal error: promotion arrays are inconsistent" >&2
  exit 3
fi

print_plan() {
  printf 'Protocol: complete-cache dev-only promotion; epochs=%s (max=3) seed=%s\n' \
    "$epochs" "$seed"
  printf 'Selection: %s; epoch-0 exact PTH base remains eligible\n' \
    "$selection_metric"
  printf 'Train cache: %s\nDev cache:   %s\nHeads root:  %s\n' \
    "$train_cache" "$dev_cache" "$heads_root"
  printf 'Concurrency: %s head(s) per GPU on GPU_IDS=%s\n' \
    "$max_heads_per_gpu" "$gpu_ids"
  printf '\n%-46s %-15s %-31s %-8s %-8s\n' \
    RUN FAMILY ARM LR DETAIL
  for index in "${!run_names[@]}"; do
    detail=cap1.5_b32
    if [[ ${families[$index]} == compact_axial ]]; then
      detail=d64_depth1_drop0
    fi
    printf '%-46s %-15s %-31s %-8s %-8s\n' \
      "${run_names[$index]}" \
      "${families[$index]}" \
      "${arms[$index]}" \
      "${learning_rates[$index]}" \
      "$detail"
  done
}

if [[ "$mode" == plan ]]; then
  print_plan
  printf '\nDry-run only. Reinvoke with --run to launch these four candidates.\n'
  exit 0
fi

for required in \
  "$python_bin" "$gated_runner" "$two_axis_runner" \
  "$train_cache" "$dev_cache"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done
if [[ -e "$heads_root" ]]; then
  echo "Refusing existing promotion output root: $heads_root" >&2
  exit 2
fi

# Audit+SHA preflight avoids loading the complete feature tensors in the
# launcher. The actual runners still validate the cache payloads themselves.
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


def exact(value):
    return os.path.abspath(os.path.expanduser(str(value)))


def read_audit(cache_value, role):
    cache = Path(cache_value).expanduser().absolute()
    audit_path = Path(str(cache) + ".audit.json")
    if not audit_path.is_file():
        raise SystemExit(f"{role} audit is missing: {audit_path}")
    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    schema = audit.get("schema_version")
    if schema == "legacy360-merged-feature-cache-audit-v1":
        output = audit.get("output", {})
        compatibility = audit.get("compatibility", {})
        feature = compatibility.get("tensor_fields", {}).get("features", {})
        item = {
            "split": audit.get("split"),
            "path": output.get("path"),
            "sha256": output.get("sha256"),
            "rows": output.get("rows"),
            "shape": feature.get("shape"),
            "encoder": compatibility.get("encoder_fingerprint"),
            "sealed_authorized": False,
            "parts": len(audit.get("inputs", [])),
            "integrity": audit.get("integrity", {}),
        }
    elif schema == "query360-two-axis-feature-cache-v1":
        encoder = audit.get("encoder")
        item = {
            "split": audit.get("split"),
            "path": audit.get("cache"),
            "sha256": audit.get("cache_sha256"),
            "rows": audit.get("rows"),
            "shape": audit.get("feature_shape"),
            "encoder": (
                fingerprint(encoder) if isinstance(encoder, dict) else None
            ),
            "sealed_authorized": bool(
                audit.get("extraction", {}).get(
                    "sealed_test_authorized", False
                )
            ),
            "parts": None,
            "integrity": {},
        }
    else:
        raise SystemExit(
            f"{role} audit schema is unsupported: {schema!r}"
        )
    if exact(item["path"]) != str(cache):
        raise SystemExit(f"{role} audit/cache path mismatch")
    if sha256_file(cache) != item["sha256"]:
        raise SystemExit(f"{role} cache SHA256 differs from audit")
    shape = item["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or int(shape[0]) != int(item["rows"])
        or int(item["rows"]) < 1
    ):
        raise SystemExit(f"{role} audit has an invalid feature shape")
    if not isinstance(item["encoder"], str):
        raise SystemExit(f"{role} audit has no encoder fingerprint")
    return item


train = read_audit(sys.argv[1], "train")
dev = read_audit(sys.argv[2], "dev")
if train["split"] not in {"train", "train_core"}:
    raise SystemExit(f"train cache split is not train_core: {train['split']!r}")
if dev["split"] not in {"dev", "evaluation"}:
    raise SystemExit(f"dev cache split is not dev: {dev['split']!r}")
if train["sealed_authorized"] or dev["sealed_authorized"]:
    raise SystemExit("development cache audit records held-out authorization")
if train["encoder"] != dev["encoder"]:
    raise SystemExit("train/dev encoder provenance differs")
if tuple(train["shape"][1:]) != tuple(dev["shape"][1:]):
    raise SystemExit("train/dev feature shapes differ")
if int(train["rows"]) != 113843:
    raise SystemExit(
        f"complete train cache must have 113843 rows, got {train['rows']}"
    )
if int(dev["rows"]) != 12621:
    raise SystemExit(
        f"canonical dev cache must have 12621 rows, got {dev['rows']}"
    )
if tuple(train["shape"][1:]) != (4, 3, 768):
    raise SystemExit(
        f"full-cache feature tail must be [4,3,768], got {train['shape']}"
    )
if train["parts"] != 15:
    raise SystemExit(
        f"complete train cache must merge 15 parts, got {train['parts']}"
    )
integrity = train["integrity"]
for key in (
    "ids_unique",
    "query360_indices_unique",
    "plumes_disjoint_between_inputs",
):
    if integrity.get(key) is not True:
        raise SystemExit(f"complete train audit failed integrity.{key}")
print(
    "full-cache SHA/audit preflight passed:",
    int(train["rows"]),
    "train rows;",
    int(dev["rows"]),
    "dev rows",
)
PY

mkdir -p "$(dirname "$heads_root")"
mkdir "$heads_root"

write_campaign_status() {
  local state=$1
  local exit_code=$2
  "$python_bin" - \
    "$heads_root/promotion_launcher_status.json" \
    "$state" "$exit_code" "$epochs" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

destination, state, exit_code, epochs = sys.argv[1:]
payload = {
    "schema_version": "legacy360-full-cache-promotion-status-v1",
    "status": state,
    "utc": datetime.now(timezone.utc).isoformat(),
    "exit_code": int(exit_code),
    "candidate_count": 4,
    "epochs": int(epochs),
    "epoch0_base_required": True,
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

write_run_status() {
  local destination=$1
  local state=$2
  local exit_code=$3
  local run_name=$4
  local family=$5
  local gpu=$6
  local arm=$7
  local learning_rate=$8
  local epoch0_verified=$9
  "$python_bin" - \
    "$destination" "$state" "$exit_code" "$run_name" "$family" "$gpu" \
    "$arm" "$learning_rate" "$epochs" "$epoch0_verified" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    destination,
    state,
    exit_code,
    run_name,
    family,
    gpu,
    arm,
    learning_rate,
    epochs,
    epoch0_verified,
) = sys.argv[1:]
payload = {
    "schema_version": "legacy360-full-cache-promotion-run-status-v1",
    "status": state,
    "utc": datetime.now(timezone.utc).isoformat(),
    "exit_code": int(exit_code),
    "run": run_name,
    "family": family,
    "gpu": int(gpu),
    "base_mode": "universal",
    "arm": arm,
    "learning_rate": float(learning_rate),
    "epochs": int(epochs),
    "epoch0_base_required": True,
    "epoch0_base_verified": epoch0_verified == "true",
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

postflight() {
  local run_dir=$1
  local family=$2
  local arm=$3
  local learning_rate=$4
  "$python_bin" - \
    "$run_dir" "$family" "$arm" "$learning_rate" "$epochs" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).absolute()
family = sys.argv[2]
arm = sys.argv[3]
learning_rate = float(sys.argv[4])
epochs = int(sys.argv[5])


def read_json(name):
    path = run_dir / name
    if not path.is_file():
        raise SystemExit(f"runner artifact is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload


status = read_json("run_status.json")
summary = read_json("summary.json")
lock = read_json("selection_lock.json")
history = read_json("metrics_history.json")
checkpoint = run_dir / "checkpoint_best.pth"
if not checkpoint.is_file():
    raise SystemExit("runner checkpoint is missing")
if status.get("status") != "complete" or summary.get("status") != "complete":
    raise SystemExit("runner/summary status is not complete")
for payload, role in ((status, "status"), (summary, "summary")):
    if payload.get("sealed_test_read") is not False:
        raise SystemExit(f"{role} records a held-out read")
    if int(payload.get("sealed_test_evaluations", -1)) != 0:
        raise SystemExit(f"{role} records a held-out evaluation")
if status.get("dev_only") not in {None, True}:
    raise SystemExit("runner status is not development-only")
if lock.get("test_cache_read_before_lock") is not False:
    raise SystemExit("selection lock lacks no-held-out-read attestation")
if summary.get("selection_lock") != lock:
    raise SystemExit("summary and selection lock differ")
if lock.get("base_mode") != "universal" or lock.get("arm") != arm:
    raise SystemExit("locked base/arm differs from promotion candidate")
locked_checkpoint = os.path.abspath(
    os.path.expanduser(str(lock.get("checkpoint", "")))
)
if locked_checkpoint != str(checkpoint):
    raise SystemExit("selection lock names the wrong checkpoint")
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if lock.get("checkpoint_sha256") != digest:
    raise SystemExit("checkpoint SHA256 differs from selection lock")
best_epoch = int(lock.get("best_epoch", -1))
if not 0 <= best_epoch <= epochs:
    raise SystemExit("locked epoch is outside the promotion budget")
if not isinstance(lock.get("encoder"), dict):
    raise SystemExit("selection lock has no encoder provenance")

if not isinstance(history, list) or not history:
    raise SystemExit("metrics history is empty")
history_epochs = [int(item.get("epoch", -1)) for item in history]
if (
    history_epochs[0] != 0
    or history[0].get("train") is not None
    or history_epochs != sorted(set(history_epochs))
    or history_epochs[-1] > epochs
):
    raise SystemExit("metrics history does not preserve eligible epoch 0")
summary_history = summary.get("history")
if (
    not isinstance(summary_history, list)
    or not summary_history
    or int(summary_history[0].get("epoch", -1)) != 0
    or summary_history[0].get("train") is not None
):
    raise SystemExit("summary does not preserve eligible epoch 0")

training = summary.get("training", {})
model = summary.get("model", {})
config = lock.get("model_config", {})
if not math.isclose(
    float(training.get("learning_rate", -1.0)),
    learning_rate,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit("summary learning rate differs")
if family == "gated_delta":
    if int(training.get("epochs_requested", -1)) != epochs:
        raise SystemExit("gated epoch budget differs")
    if (
        int(config.get("bottleneck_dim", -1)) != 32
        or not math.isclose(
            float(config.get("residual_cap", -1.0)),
            1.5,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SystemExit("gated compact config differs")
elif family == "compact_axial":
    if int(training.get("epochs", -1)) != epochs:
        raise SystemExit("axial epoch budget differs")
    if (
        int(config.get("model_dim", -1)) != 64
        or int(config.get("num_heads", -1)) != 4
        or int(config.get("temporal_depth", -1)) != 1
        or not math.isclose(
            float(config.get("dropout", -1.0)),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SystemExit("compact axial config differs")
else:
    raise SystemExit(f"unsupported family: {family}")
if model.get("base_mode") != "universal" or model.get("arm") != arm:
    raise SystemExit("summary base/arm differs")
PY
}

run_one() {
  local index=$1
  local gpu=$2
  local run_name=${run_names[$index]}
  local family=${families[$index]}
  local arm=${arms[$index]}
  local learning_rate=${learning_rates[$index]}
  local residual_cap=${residual_caps[$index]}
  local run_dir=${heads_root}/${run_name}
  local log_path=${run_dir}/stdout_stderr.log
  local launch_status=${run_dir}/launcher_status.json
  local -a command

  mkdir "$run_dir"
  write_run_status \
    "$launch_status" running 0 "$run_name" "$family" "$gpu" \
    "$arm" "$learning_rate" false

  if [[ "$family" == gated_delta ]]; then
    command=(
      "$python_bin" "$gated_runner" train
      --train-cache "$train_cache"
      --dev-cache "$dev_cache"
      --output-dir "$run_dir"
      --base-mode universal
      --arm "$arm"
      --epochs "$epochs"
      --early-stop-patience 2
      --seed "$seed"
      --batch-size "$batch_size"
      --eval-batch-size "$eval_batch_size"
      --learning-rate "$learning_rate"
      --weight-decay 0.01
      --sensor-aux-weight 0.1
      --residual-l2 1e-3
      --grad-clip 1.0
      --bottleneck-dim 32
      --dropout 0.05
      --residual-cap "$residual_cap"
      --selection-metric "$selection_metric"
      --device "cuda:${gpu}"
    )
  else
    command=(
      "$python_bin" "$two_axis_runner" train
      --train-cache "$train_cache"
      --dev-cache "$dev_cache"
      --output-dir "$run_dir"
      --base-mode universal
      --arm "$arm"
      --epochs "$epochs"
      --seed "$seed"
      --batch-size "$batch_size"
      --eval-batch-size "$eval_batch_size"
      --learning-rate "$learning_rate"
      --weight-decay 0.01
      --sensor-aux-weight 0.05
      --axis-aux-weight 0.05
      --grad-clip 1.0
      --model-dim 64
      --num-heads 4
      --temporal-depth 1
      --mlp-ratio 1.0
      --dropout 0.0
      --selection-metric "$selection_metric"
      --device "cuda:${gpu}"
    )
  fi

  {
    printf '[promotion] run=%s family=%s gpu=%s arm=%s lr=%s\n' \
      "$run_name" "$family" "$gpu" "$arm" "$learning_rate"
    printf '[promotion] train_cache=%s\n[promotion] dev_cache=%s\n' \
      "$train_cache" "$dev_cache"
    printf '[promotion] epoch0 exact-PTH base is selection-eligible\n'
    printf '[promotion] command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    set +e
    "${command[@]}"
    local exit_code=$?
    set -e

    local final_state=failed
    local epoch0_verified=false
    if [[ $exit_code -eq 0 ]] &&
       postflight \
         "$run_dir" "$family" "$arm" "$learning_rate"; then
      final_state=complete
      epoch0_verified=true
    elif [[ $exit_code -eq 0 ]]; then
      exit_code=3
    fi
    write_run_status \
      "$launch_status" "$final_state" "$exit_code" "$run_name" \
      "$family" "$gpu" "$arm" "$learning_rate" "$epoch0_verified"
    printf '[promotion] final_state=%s exit_code=%s epoch0_verified=%s\n' \
      "$final_state" "$exit_code" "$epoch0_verified"
    return "$exit_code"
  } >"$log_path" 2>&1
}

write_campaign_status running 0
print_plan
printf '\nLaunching four complete-cache development-only candidates.\n'

# Round-robin physical GPUs before adding a second process per GPU.
slots=()
for ((slot = 0; slot < max_heads_per_gpu; slot += 1)); do
  for gpu in "${gpu_array[@]}"; do
    slots+=("$gpu")
  done
done

overall_exit=0
for ((
  wave_start = 0;
  wave_start < ${#run_names[@]};
  wave_start += ${#slots[@]}
)); do
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
      echo "Promotion candidate failed: ${wave_names[$pid_index]}" >&2
      overall_exit=1
    fi
  done
done

if [[ $overall_exit -ne 0 ]]; then
  write_campaign_status failed "$overall_exit"
  echo "Promotion finished with failures; inspect per-run logs/status." >&2
  exit "$overall_exit"
fi
write_campaign_status complete 0
echo "All 4 complete-cache development-only promotion candidates completed."
