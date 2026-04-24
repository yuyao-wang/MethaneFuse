#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

TMP_BASE=${TMP_BASE:-/transferdiniu2/yuyao/temp}
mkdir -p "$TMP_BASE"
chmod 700 "$TMP_BASE"
export TMPDIR="$TMP_BASE"
export TMP="$TMP_BASE"
export TEMP="$TMP_BASE"

DATA_ROOT=${DATA_ROOT:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m}
TRAIN_CSV=${TRAIN_CSV:-$DATA_ROOT/manifest_time_train.csv}
TEST_CSV=${TEST_CSV:-$DATA_ROOT/manifest_time_test.csv}
OUT_ROOT=${OUT_ROOT:-/transferdiniu2/yuyao/checkpoints/unet_multisensor_baseline_iou_plus}
CACHE_DIR=${CACHE_DIR:-/diniuvol/yuyao/local_train_temp_cache_480m}
RUN_LABEL=${RUN_LABEL:-480m_unet_baseline_iou_plus_full}
PYTHON_BIN=${PYTHON_BIN:-python}

[ -f "$TRAIN_CSV" ] || { echo "Missing TRAIN_CSV: $TRAIN_CSV" >&2; exit 1; }
[ -f "$TEST_CSV" ] || { echo "Missing TEST_CSV: $TEST_CSV" >&2; exit 1; }

"$PYTHON_BIN" universal_models_fusion/unet_multisensor_baseline_iou_plus.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --tasks s2,l89,emit \
  --epochs 7 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_workers 8 \
  --device cuda \
  --log_interval 100 \
  --checkpoint_dir "$OUT_ROOT" \
  --use_wandb \
  --wandb_project "query_dataset" \
  --wandb_run_name "$RUN_LABEL" \
  --local_cache_dir "$CACHE_DIR" \
  --local_cache_min_free_gb 100 
