#!/usr/bin/env bash
set -uo pipefail

cd /home/yuyao/panopticon

GPUS_CSV="${1:-0,1}"
MAX_PARALLEL="${2:-4}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if (( ${#GPUS[@]} == 0 )); then
  echo "No GPUs provided. Usage: bash run_single_weight_sweep.sh 0,1 4"
  exit 1
fi
if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}"
  exit 1
fi

mkdir -p /transferdiniu2/yuyao/temp
chmod 700 /transferdiniu2/yuyao/temp
export TMPDIR=/transferdiniu2/yuyao/temp
export TMP=/transferdiniu2/yuyao/temp
export TEMP=/transferdiniu2/yuyao/temp

LOG_DIR="/home/yuyao/panopticon/logs/single_weight_sweep"
mkdir -p "${LOG_DIR}"

SCRIPT="universal_models_fusion/multi_sensor_panopticon_4_overall_boost.py"

COMMON_ARGS=(
  --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv
  --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv
  --weights weights/panopticon_vitb14_teacher.pth
  --local_cache_dir /diniuvol/yuyao/local_train_temp_cache
  --local_cache_min_free_gb 200
  --train_backbone
  --freeze_backbone_epochs 1
  --batch_size 32
  --epochs 8
  --head_lr 1e-3
  --backbone_lr 5e-5
  --weight_decay 5e-4
  --lr_scheduler noam
  --warmup_steps 4000
  --num_workers 8
  --device cuda
  --row_fusion_mode map
  --row_gate
  --non_overlap_kd_weight 0.0
  --gate_supervision_weight 0.02
  --sensor_aux_loss_weight 0.3
  --single_fused_ce_weight 0.5
  --best_non_overlap_tolerance 0.002
  --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_overall
  --use_wandb
  --wandb_project baselines
)

run_one() {
  local gpu_id="$1"
  local run_name="$2"
  shift
  shift
  local ts
  ts="$(date -u +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${ts}_${run_name}.log"

  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "[$(date -u +%F' '%T)] START ${run_name} (GPU=${gpu_id})"
    python "${SCRIPT}" \
      "${COMMON_ARGS[@]}" \
      "$@" \
      --wandb_run_name "${run_name}" 2>&1 | tee "${log_file}"
    echo "[$(date -u +%F' '%T)] END   ${run_name} (GPU=${gpu_id})"
    echo "log: ${log_file}"
  )
}

declare -a RUN_SPECS=(
  "ovb_singlew_ctrl_all1|--single_weight_s2 1.0 --single_weight_l89 1.0 --single_weight_s5p 1.0 --single_weight_wv3 1.0"
  "ovb_singlew_wv3_1p5|--single_weight_s2 1.0 --single_weight_l89 1.0 --single_weight_s5p 1.0 --single_weight_wv3 1.5"
  "ovb_singlew_wv3_2p0|--single_weight_s2 1.0 --single_weight_l89 1.0 --single_weight_s5p 1.0 --single_weight_wv3 2.0"
  "ovb_singlew_wv3_2p5|--single_weight_s2 1.0 --single_weight_l89 1.0 --single_weight_s5p 1.0 --single_weight_wv3 2.5"
  "ovb_singlew_s5p_1p2|--single_weight_s2 1.0 --single_weight_l89 1.0 --single_weight_s5p 1.2 --single_weight_wv3 1.0"
)

echo "GPUs=${GPUS_CSV}, MAX_PARALLEL=${MAX_PARALLEL}, total_runs=${#RUN_SPECS[@]}"

declare -a PIDS=()
FAILED=0
launch_idx=0

for spec in "${RUN_SPECS[@]}"; do
  run_name="${spec%%|*}"
  run_args="${spec#*|}"
  gpu_id="${GPUS[$((launch_idx % ${#GPUS[@]}))]}"

  # shellcheck disable=SC2086
  run_one "${gpu_id}" "${run_name}" ${run_args} &
  PIDS+=("$!")
  launch_idx=$((launch_idx + 1))

  if (( ${#PIDS[@]} >= MAX_PARALLEL )); then
    wait -n || FAILED=1
    live=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        live+=("${pid}")
      fi
    done
    PIDS=("${live[@]}")
  fi
done

for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAILED=1
done

if (( FAILED != 0 )); then
  echo "Some runs failed."
  exit 1
fi
echo "All sweep runs finished successfully."
