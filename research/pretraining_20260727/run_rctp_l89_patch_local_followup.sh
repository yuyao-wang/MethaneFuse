#!/usr/bin/env bash
set -euo pipefail

# Train/development-only frozen final-patch diagnostic.
#
# Resource contract (A100 80 GiB):
#   * one extraction process at a time on exactly DEVICE;
#   * batch=12, workers=4, prefetch=1;
#   * refuse launch below 30 GiB free; expected 5-8 GiB, measured cap 12 GiB;
#   * no raw remote I/O: every image must already be below REQUIRED_LOCAL_ROOT;
#   * local residual heads run on CPU after every selected arm is cached.
#
# Do not launch until the owner of the selected physical GPU has agreed.

repo_root=${REPO_ROOT:-/home/yuyao/panopticon}
python_bin=${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}
device=${DEVICE:-cuda:0}
output_root=${OUTPUT_ROOT:-/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_patch_local_followup_v1}
data_root=${DATA_ROOT:-/diniuvol/yuyao/methanefuse_research_20260727}
manifest_root=${MANIFEST_ROOT:-$data_root/manifests_staged/l89_6time}
baseline_root=${BASELINE_ROOT:-$data_root/rctp_l89_real_cls_v1/downstream_role_only_seed20260728}
rctp_root=${RCTP_ROOT:-$data_root/rctp_l89_real_cls_v1}
local_root=${REQUIRED_LOCAL_ROOT:-$data_root/cache}
batch_size=${BATCH_SIZE:-12}
num_workers=${NUM_WORKERS:-4}
prefetch_factor=${PREFETCH_FACTOR:-1}
arms_csv=${ARMS:-p0,p4,p5}

script=research/pretraining_20260727/l89_patch_local_rctp_followup.py
test_script=research/pretraining_20260727/test_l89_patch_local_rctp_followup_cpu.py
launcher_source=$(realpath "$0")
p0_weights=/home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth
p4_weights=$rctp_root/pretrain/p4/backbone_best_dev_ap.pth
p5_weights=$rctp_root/pretrain/p5/backbone_best_dev_ap.pth
p0_cache_root=$data_root/cache/l89_ragged_cls_v1
p4_cache_root=$rctp_root/cache/p4
p5_cache_root=$rctp_root/cache/p5

cd "$repo_root"
export REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output_root/cache" "$output_root/logs" "$output_root/provenance"

# Freeze the exact executable source before the first GPU process.  All child
# commands use the immutable copy, and both source and snapshot hashes are
# checked again after the comparison completes.
sha256sum \
  "$repo_root/$script" \
  "$repo_root/$test_script" \
  "$launcher_source" \
  > "$output_root/provenance/source_before.sha256"
cp "$repo_root/$script" "$output_root/provenance/l89_patch_local_rctp_followup.py"
cp "$repo_root/$test_script" "$output_root/provenance/test_l89_patch_local_rctp_followup_cpu.py"
cp "$launcher_source" "$output_root/provenance/run_rctp_l89_patch_local_followup.sh"
sha256sum \
  "$output_root/provenance/l89_patch_local_rctp_followup.py" \
  "$output_root/provenance/test_l89_patch_local_rctp_followup_cpu.py" \
  "$output_root/provenance/run_rctp_l89_patch_local_followup.sh" \
  > "$output_root/provenance/snapshot.sha256"
runtime_script=$output_root/provenance/l89_patch_local_rctp_followup.py

IFS=',' read -r -a selected_arms <<< "$arms_csv"
train_args=()
for arm in "${selected_arms[@]}"; do
  case "$arm" in
    p0)
      weights=$p0_weights
      cls_root=$p0_cache_root
      ;;
    p4)
      weights=$p4_weights
      cls_root=$p4_cache_root
      ;;
    p5)
      weights=$p5_weights
      cls_root=$p5_cache_root
      ;;
    *)
      echo "Unknown arm: $arm" >&2
      exit 2
      ;;
  esac
  head_checkpoint=$baseline_root/$arm/role_only/checkpoint_best_ap.pt
  for split in train val; do
    output_cache=$output_root/cache/${arm}_${split}.pt
    predictions_args=()
    if [[ "$split" == val ]]; then
      predictions_args=(
        --base-validation-predictions
        "$baseline_root/$arm/role_only/validation_best_ap_predictions.csv"
      )
    fi
    if [[ ! -f "$output_cache" ]]; then
      "$python_bin" "$runtime_script" extract \
        --arm "$arm" \
        --split "$split" \
        --csv "$manifest_root/$split.csv" \
        --cls-cache "$cls_root/$split.pt" \
        --weights "$weights" \
        --base-head-checkpoint "$head_checkpoint" \
        "${predictions_args[@]}" \
        --output-cache "$output_cache" \
        --device "$device" \
        --batch-size "$batch_size" \
        --num-workers "$num_workers" \
        --prefetch-factor "$prefetch_factor" \
        --amp-dtype float16 \
        --projection-dim 64 \
        --projection-seed 36064 \
        --topk-fraction 0.10 \
        --min-cuda-free-gib 30 \
        --max-cuda-allocated-gib 12 \
        --required-local-root "$local_root" \
        2>&1 | tee "$output_root/logs/${arm}_${split}_extract.log"
    fi
  done
  train_args+=(--arm "$arm" "$output_root/cache/${arm}_train.pt" "$output_root/cache/${arm}_val.pt")
done

"$python_bin" "$runtime_script" train-compare \
  "${train_args[@]}" \
  --output-dir "$output_root/comparison_seed20260728" \
  --device cpu \
  --epochs 3 \
  --batch-size 256 \
  --learning-rate 3e-3 \
  --weight-decay 0 \
  --residual-cap 1.5 \
  --seed 20260728 \
  2>&1 | tee "$output_root/logs/train_compare.log"

sha256sum -c "$output_root/provenance/source_before.sha256"
sha256sum -c "$output_root/provenance/snapshot.sha256"
