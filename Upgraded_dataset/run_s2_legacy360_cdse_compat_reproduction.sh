#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/yuyao/panopticon"
PYTHON="/home/yuyao/miniconda3/envs/panopticon/bin/python"
TRAIN_SCRIPT="${REPO_ROOT}/Upgraded_dataset/dino_classifier_head_s2_legacy360_repro.py"
SPLIT_ROOT="${REPO_ROOT}/Upgraded_dataset/s2_legacy360_matched_gee_splits/matched_gee_crops36_splits"
RESULT_ROOT="${REPO_ROOT}/Upgraded_dataset/s2_legacy360_matched_gee_splits/reproduction_results"
CHECKPOINT_ROOT="/transferdiniu2/yuyao/checkpoints/s2_legacy360_matched_gee_cdse_compat_20260728"
WEIGHTS="${REPO_ROOT}/weights/panopticon_vitb14_teacher.pth"
GPU_INDEX="${GPU_INDEX:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${RESULT_ROOT}" "${CHECKPOINT_ROOT}"

wait_for_gpu() {
  while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX}" | tr -d ' ')"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      echo "[GPU guard] GPU${GPU_INDEX} free=${free_mib} MiB; starting."
      return
    fi
    echo "[GPU guard] GPU${GPU_INDEX} free=${free_mib:-unknown} MiB; waiting for >=${MIN_FREE_MIB} MiB."
    sleep 60
  done
}

run_split() {
  local split_name="$1"
  local train_csv="${SPLIT_ROOT}/${split_name}/train.csv"
  local test_csv="${SPLIT_ROOT}/${split_name}/test.csv"
  local run_name="s2_legacy360_matched_gee_cdse_compat_${split_name}_full_finetune"
  local log_path="${RESULT_ROOT}/${split_name}.cdse_compat.log"
  local metrics_path="${RESULT_ROOT}/${split_name}.cdse_compat.metrics.jsonl"

  test -s "${train_csv}"
  test -s "${test_csv}"
  rm -f "${metrics_path}"
  wait_for_gpu

  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON}" "${TRAIN_SCRIPT}" \
    --train_csv "${train_csv}" \
    --test_csv "${test_csv}" \
    --weights "${WEIGHTS}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --head_lr 0.001 \
    --backbone_lr 0.0001 \
    --lr_scheduler noam \
    --warmup_steps 4000 \
    --weight_decay 0.0005 \
    --momentum 0.9 \
    --num_workers "${NUM_WORKERS}" \
    --pad_to_multiple 1 \
    --input_resize_size 224 \
    --legacy_t0_cdse_contract \
    --device cuda \
    --log_interval 100 \
    --t0_col s2_0_path \
    --t90_col s2_90_path \
    --t360_col s2_360_path \
    --train_backbone \
    --freeze_backbone_epochs 0 \
    --checkpoint_dir "${CHECKPOINT_ROOT}" \
    --run_name "${run_name}" \
    --metrics_jsonl "${metrics_path}" \
    2>&1 | tee "${log_path}"
}

cd "${REPO_ROOT}"
run_split "row_random_80_20"
run_split "event_disjoint_80_20"
