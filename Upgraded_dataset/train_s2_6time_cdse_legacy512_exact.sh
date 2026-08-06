#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yuyao/miniconda3/envs/panopticon/bin/python}"
PANOPTICON_ROOT="${PANOPTICON_ROOT:-/home/yuyao/panopticon}"
DATA_ROOT="${DATA_ROOT:-/diniuvol/yuyao/s2_6time_cdse_legacy512_exact_cache/input32}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/train_patches_32_local.csv}"
TEST_CSV="${TEST_CSV:-${DATA_ROOT}/test_patches_32_local.csv}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-40000}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  while true; do
    read -r GPU_INDEX GPU_FREE_MIB < <(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}' \
        | sort -k2,2nr \
        | head -n 1
    )
    if [[ -n "${GPU_INDEX:-}" && "${GPU_FREE_MIB:-0}" -ge "${MIN_GPU_FREE_MIB}" ]]; then
      export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
      echo "[GPU] selected physical GPU ${GPU_INDEX} with ${GPU_FREE_MIB} MiB free"
      break
    fi
    echo "[GPU] waiting for >=${MIN_GPU_FREE_MIB} MiB free; best=${GPU_FREE_MIB:-unknown} MiB"
    sleep 60
  done
fi

cd "${PANOPTICON_ROOT}"

exec "${PYTHON}" \
  Upgraded_dataset/dino_classifier_head_s2_temporal_satmae.py \
  --train_csv "${TRAIN_CSV}" \
  --test_csv "${TEST_CSV}" \
  --weights /home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth \
  --train_backbone \
  --channel_indices all \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --epochs 50 \
  --num_workers 16 \
  --input_resize_size 224 \
  --stats_samples 2000 \
  --stats_workers 8 \
  --local_cache_mode off \
  --checkpoint_dir /transferdiniu2/yuyao/checkpoints/s2_cdse_legacy512_exact \
  --run_name s2_6time_cdse_legacy512_exact_all12_full_finetune \
  "$@"
