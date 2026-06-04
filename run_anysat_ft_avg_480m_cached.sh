#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-/diniuvol/yuyao/tmp}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}"
RUN_NAME="${RUN_NAME:-anysat_ft_avg_480m_full_20260525}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/transferdiniu2/yuyao/checkpoints/anysat_ft_avg_480m}"
CACHE_DIR="${PROCESSED_CACHE_DIR:-/diniuvol/yuyao/anysat_ft_avg_480m_processed_cache}"

exec "${PYTHON_BIN}" baseline/anysat_ft_avg_fusion_480m.py \
  --run_name "${RUN_NAME}" \
  --train_csv "${TRAIN_CSV:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m_518/manifest_time_train.csv}" \
  --test_csv "${TEST_CSV:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m_518/manifest_time_test.csv}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-12}" \
  --epochs "${EPOCHS:-7}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --device "${DEVICE:-cuda}" \
  --lr_anysat "${LR_ANYSAT:-1e-5}" \
  --lr_head "${LR_HEAD:-1e-3}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.05}" \
  --patch_size "${PATCH_SIZE:-20}" \
  --freeze_anysat_epochs "${FREEZE_ANYSAT_EPOCHS:-1}" \
  --log_interval "${LOG_INTERVAL:-100}" \
  --processed_cache_dir "${CACHE_DIR}" \
  --processed_cache_min_free_gb "${PROCESSED_CACHE_MIN_FREE_GB:-100}" \
  --resume \
  --amp \
  "$@"
