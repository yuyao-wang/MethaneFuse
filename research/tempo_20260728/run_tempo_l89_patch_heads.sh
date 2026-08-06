#!/usr/bin/env bash
set -euo pipefail

# Launch one or more train/dev-only TEMPO heads as independent user services.
# Examples:
#   ARMS=p5_a0 ./research/tempo_20260728/run_tempo_l89_patch_heads.sh
#   ARMS=p5_a1,p5_a2,p0_a0,p0_a1 \
#     ./research/tempo_20260728/run_tempo_l89_patch_heads.sh

repo_root=${REPO_ROOT:-/home/yuyao/panopticon}
python_bin=${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}
run_root=${RUN_ROOT:-/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_v1}
device=${DEVICE:-cuda:0}
physical_gpu=${PHYSICAL_GPU:-0}
arms_csv=${ARMS:-p5_a0,p5_a1,p5_a2,p0_a0,p0_a1}
seed=${SEED:-20260728}
arm_output_suffix=${ARM_OUTPUT_SUFFIX:-}
zero_init_mode=${ZERO_INIT_MODE:-scalar}
script=$repo_root/research/tempo_20260728/tempo_l89_patch.py
train_manifest=$run_root/cache/train/manifest.json
val_manifest=$run_root/cache/val/manifest.json

cd "$repo_root"
mkdir -p "$run_root/heads" "$run_root/logs" "$run_root/provenance"

for required in \
  "$script" \
  "$train_manifest" \
  "$val_manifest" \
  "$run_root/base_overlays/event_p0_train.pt" \
  "$run_root/base_overlays/event_p0_val.pt" \
  "$run_root/base_overlays/event_p5_train.pt" \
  "$run_root/base_overlays/event_p5_val.pt"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 2
  fi
done

sha256sum \
  "$script" \
  "$repo_root/research/tempo_20260728/test_tempo_l89_patch_cpu.py" \
  "$repo_root/research/tempo_20260728/PATCH_PROTOCOL.md" \
  "$repo_root/research/tempo_20260728/run_tempo_l89_patch_heads.sh" \
  > "$run_root/provenance/head_source_before.sha256"

IFS=',' read -r -a selected_arms <<< "$arms_csv"
for arm in "${selected_arms[@]}"; do
  topk_fraction=0.10
  radius=1
  case "$arm" in
    p5_a0)
      family=p5
      normality_args=(--no-use-normality-features --normality-scale 0)
      null_weight=0
      ;;
    p5_a1)
      family=p5
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0
      ;;
    p5_a2)
      family=p5
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0.1
      ;;
    p0_a0)
      family=p0
      normality_args=(--no-use-normality-features --normality-scale 0)
      null_weight=0
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    p0_a1)
      family=p0
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    p0_a2)
      family=p0
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0.1
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    p0_a1_r0)
      family=p0
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0
      radius=0
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    p0_a1_r2)
      family=p0
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0
      radius=2
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    p0_a1_k02)
      family=p0
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0
      topk_fraction=0.02
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    p0_a1_k05)
      family=p0
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0
      topk_fraction=0.05
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    p0_a1_k05_r0)
      family=p0
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0
      topk_fraction=0.05
      radius=0
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    p0_a1_k05_r2)
      family=p0
      normality_args=(--use-normality-features --normality-scale 1)
      null_weight=0
      topk_fraction=0.05
      radius=2
      late_fusion_args=(
        --val-late-fusion-overlay
        "$run_root/base_overlays/event_p5_val.pt"
      )
      ;;
    *)
      echo "Unknown TEMPO arm: $arm" >&2
      exit 2
      ;;
  esac
  if [[ "$family" == p5 ]]; then
    late_fusion_args=()
  fi
  output_dir=$run_root/heads/${arm}${arm_output_suffix}
  log_path=$run_root/logs/${arm}${arm_output_suffix}.log
  unit=tempo-l89-patch-${arm//_/-}${arm_output_suffix//_/-}-v1
  if systemctl --user is-active --quiet "$unit.service"; then
    echo "$unit.service is already active" >&2
    exit 3
  fi
  mkdir -p "$output_dir"
  systemd-run \
    --user \
    --unit="$unit" \
    --collect \
    --working-directory="$repo_root" \
    --setenv="CUDA_VISIBLE_DEVICES=$physical_gpu" \
    --setenv="OMP_NUM_THREADS=2" \
    --setenv="MKL_NUM_THREADS=2" \
    --setenv="PYTHONPATH=$repo_root" \
    --property="StandardOutput=append:$log_path" \
    --property="StandardError=append:$log_path" \
    "$python_bin" "$script" train \
      --train-manifest "$train_manifest" \
      --val-manifest "$val_manifest" \
      --train-base-overlay "$run_root/base_overlays/event_${family}_train.pt" \
      --val-base-overlay "$run_root/base_overlays/event_${family}_val.pt" \
      "${late_fusion_args[@]}" \
      --output-dir "$output_dir" \
      --device "$device" \
      --match-rank 16 \
      --value-dim 32 \
      --hidden-dim 64 \
      --radius "$radius" \
      --temperature 0.10 \
      --topk-fraction "$topk_fraction" \
      "${normality_args[@]}" \
      --residual-cap 1.5 \
      --zero-init-mode "$zero_init_mode" \
      --null-weight "$null_weight" \
      --null-margin 0 \
      --epochs 3 \
      --batch-size 16 \
      --eval-batch-size 24 \
      --learning-rate 3e-4 \
      --weight-decay 1e-4 \
      --grad-clip 1 \
      --patience 1 \
      --min-delta 2e-4 \
      --selection-metric event_balanced_ap \
      --seed "$seed" \
      --resume
  echo "launched arm=$arm unit=$unit.service log=$log_path"
done
