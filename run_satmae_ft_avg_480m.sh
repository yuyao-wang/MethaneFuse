#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-/diniuvol/yuyao/tmp}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}"
RUN_NAME="${RUN_NAME:-satmae_ft_avg_480m_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/transferdiniu2/yuyao/checkpoints/satmae_ft_avg_480m/logs}"
mkdir -p "${LOG_DIR}"

exec "${PYTHON_BIN}" baseline/satmae_ft_avg_fusion_480m.py \
  --run_name "${RUN_NAME}" \
  --train_csv "${TRAIN_CSV:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m_518/manifest_time_train.csv}" \
  --test_csv "${TEST_CSV:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m_518/manifest_time_test.csv}" \
  --satmae_repo "${SATMAE_REPO:-/diniuvol/yuyao/SatMAE}" \
  --pretrained "${SATMAE_PRETRAINED:-/diniuvol/yuyao/satmae_weights/satmae-vitbase-multispec-pretrain.safetensors}" \
  --checkpoint_dir "${CHECKPOINT_DIR:-/transferdiniu2/yuyao/checkpoints/satmae_ft_avg_480m}" \
  --batch_size "${BATCH_SIZE:-6}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-10}" \
  --accum_steps "${ACCUM_STEPS:-2}" \
  --epochs "${EPOCHS:-7}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --lr_satmae "${LR_SATMAE:-1e-5}" \
  --lr_head "${LR_HEAD:-1e-3}" \
  --freeze_satmae_epochs "${FREEZE_SATMAE_EPOCHS:-1}" \
  --local_cache_dir "${LOCAL_CACHE_DIR:-}" \
  --local_cache_max_gb "${LOCAL_CACHE_MAX_GB:-350}" \
  --local_cache_min_free_gb "${LOCAL_CACHE_MIN_FREE_GB:-100}" \
  --log_interval "${LOG_INTERVAL:-100}" \
  --device "${DEVICE:-cuda}" \
  --amp \
  "$@"
