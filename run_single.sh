#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate panopticon
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
TMP_BASE=${TMP_BASE:-/transferdiniu2/yuyao/temp}
mkdir -p "$TMP_BASE"
chmod 700 "$TMP_BASE"
export TMPDIR="$TMP_BASE"
export TMP="$TMP_BASE"
export TEMP="$TMP_BASE"

# python examples/dino_clssifier_head_s2_temportal_one_block.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_s2_geo.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_s2_geo.csv \
#     --wandb_project baselines \
#     --wandb_run_name "S2_geo_split" \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18 \
#     --local_cache_min_free_gb 50 

# python examples/dino_clssifier_head_l89_temportal_one_block.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_l89_geo.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_l89_geo.csv \
#     --wandb_project baselines \
#     --wandb_run_name "L89_geo_split" \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18

# python examples/dino_classifier_head_s5p_temporal_one_block.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_s5p_geo.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_s5p_geo.csv \
#     --wandb_project baselines \
#     --wandb_run_name "S5p_geo_split" \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --local_cache_dir /transferdiniu2/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18 \
#     --local_cache_min_free_gb 50 \
#     --batch_size 32

# /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_wv3_geo.csv
# python universal_models_fusion/dino_clssifier_head_EMIT_simulated_wv3_temporal_one_block.py \
#     --train_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_train_geo_resplit.csv \
#     --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_geo_resplit.csv \
#     --wandb_project baselines \
#     --wandb_run_name "Geo_wv3_report" \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18 \
#     --local_cache_min_free_gb 200 \
#     --batch_size 32 \
#     --epoch 10

# python universal_models_fusion/dino_classifier_head_s5p_temporal_one_block.py \
#     --train_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_train_geo_resplit.csv \
#     --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_geo_resplit.csv \
#     --wandb_project baselines \
#     --wandb_run_name "Geo_s5p_report" \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18 \
#     --local_cache_min_free_gb 200 \
#     --batch_size 32 \
#     --epoch 10


TS=$(date -u +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-/transferdiniu2/yuyao/checkpoints/480m_single4_retrain_${TS}}
mkdir -p "$OUT_ROOT"/{s2,l89,s5p,wv3,logs}

DATA_ROOT=${DATA_ROOT:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m}
TRAIN_CSV=${TRAIN_CSV:-$DATA_ROOT/manifest_time_train.csv}
TEST_CSV=${TEST_CSV:-$DATA_ROOT/manifest_time_test.csv}
INFER_CSV=${INFER_CSV:-$TEST_CSV}
RUN_LABEL=${RUN_LABEL:-query dataset 480m}
# S5P_TRAIN_CSV=/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid_emit_binary_mask_cleaned_train.csv
# S5P_TEST_CSV=/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid_emit_binary_mask_cleaned_test.csv

PYTHON_BIN=${PYTHON_BIN:-python}
WEIGHTS=${WEIGHTS:-/home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth}
CACHE_DIR=${CACHE_DIR:-/diniuvol/yuyao/local_train_temp_cache_480m}

[ -f "$TRAIN_CSV" ] || { echo "Missing TRAIN_CSV: $TRAIN_CSV" >&2; exit 1; }
[ -f "$TEST_CSV" ] || { echo "Missing TEST_CSV: $TEST_CSV" >&2; exit 1; }
# [ -f "$S5P_TRAIN_CSV" ] || { echo "Missing S5P_TRAIN_CSV: $S5P_TRAIN_CSV" >&2; exit 1; }
# [ -f "$S5P_TEST_CSV" ] || { echo "Missing S5P_TEST_CSV: $S5P_TEST_CSV" >&2; exit 1; }
[ -f "$WEIGHTS" ] || { echo "Missing WEIGHTS: $WEIGHTS" >&2; exit 1; }

run_train() {
  local name="$1"
  local gpu="$2"
  shift 2
  echo "[RUN][$name] GPU=$gpu"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$@" 2>&1 | awk 'tolower($0) !~ /warning/ { print; fflush() }' | tee "$OUT_ROOT/logs/${name}.log"
  )
}

wait_jobs() {
  local rc=0
  local pid name
  while [ "$#" -gt 0 ]; do
    pid="$1"
    name="$2"
    shift 2
    if wait "$pid"; then
      echo "[OK] $name finished"
    else
      echo "[FAIL] $name failed (pid=$pid)" >&2
      rc=1
    fi
  done
  return "$rc"
}

# Wave 1: S2 + L89 in parallel on 2 GPUs.
run_train s2 0 "$PYTHON_BIN" universal_models_fusion/dino_clssifier_head_s2_temportal_one_block.py \
  --train_csv "$TRAIN_CSV" --test_csv "$TEST_CSV" --weights "$WEIGHTS" \
  --device cuda --train_backbone --epochs 2 --batch_size 32 --num_workers 6 \
  --head_lr 1e-3 --backbone_lr 1e-4 \
  --checkpoint_dir "$OUT_ROOT/s2" --best_ckpt_path "$OUT_ROOT/s2/ckpt_best_test.pth" \
  --local_cache_dir "$CACHE_DIR" --local_cache_workers 8 --use_wandb \
  --wandb_project query_dataset \
  --wandb_run_name "S2 ${RUN_LABEL}" &
PID_S2=$!

run_train l89 1 "$PYTHON_BIN" universal_models_fusion/dino_clssifier_head_l89_temportal_one_block.py \
  --train_csv "$TRAIN_CSV" --test_csv "$TEST_CSV" --weights "$WEIGHTS" \
  --device cuda --train_backbone --epochs 7 --batch_size 32 --num_workers 6 \
  --head_lr 5e-4 --backbone_lr 5e-5 --lr_scheduler none \
  --checkpoint_dir "$OUT_ROOT/l89" --best_ckpt_path "$OUT_ROOT/l89/ckpt_best_test.pth" \
  --sensor_column anchor_sensor \
  --local_cache_dir "$CACHE_DIR" --local_cache_workers 8 --use_wandb \
  --wandb_project query_dataset \
  --wandb_run_name "L89 ${RUN_LABEL}" &
PID_L89=$!

wait_jobs "$PID_S2" s2 "$PID_L89" l89

# Wave 2: S5P + WV3 in parallel on 2 GPUs.
run_train s5p 0 "$PYTHON_BIN" universal_models_fusion/dino_classifier_head_s5p_temporal_one_block.py \
  --train_csv "$TRAIN_CSV" --test_csv "$TEST_CSV" --weights "$WEIGHTS" \
  --device cuda --train_backbone --epochs 7 --batch_size 32 --num_workers 6 \
  --head_lr 5e-4 --backbone_lr 5e-5 --lr_scheduler none \
  --balance_mode sampler \
  --best_metric test_auroc \
  --eval_threshold_mode youden \
  --best_ckpt_path "$OUT_ROOT/s5p/ckpt_best_test.pth" \
  --local_cache_dir "$CACHE_DIR" --local_cache_workers 8 --local_cache_min_free_gb 200 --use_wandb \
  --wandb_project query_dataset \
  --wandb_run_name "S5P ${RUN_LABEL}" &
PID_S5P=$!

run_train wv3 1 "$PYTHON_BIN" universal_models_fusion/dino_clssifier_head_EMIT_simulated_wv3_temporal_one_block.py \
  --train_csv "$TRAIN_CSV" --test_csv "$TEST_CSV" --weights "$WEIGHTS" \
  --device cuda --train_backbone --epochs 7 --batch_size 32 --num_workers 6 \
  --head_lr 5e-4 --backbone_lr 5e-5 --lr_scheduler none \
  --checkpoint_dir "$OUT_ROOT/wv3" --best_ckpt_path "$OUT_ROOT/wv3/ckpt_best_test.pth" \
  --sensor_column anchor_sensor \
  --local_cache_dir "$CACHE_DIR" --local_cache_workers 8 --local_cache_min_free_gb 200 --use_wandb \
  --wandb_project query_dataset \
  --wandb_run_name "WV3 ${RUN_LABEL}" &
PID_WV3=$!

wait_jobs "$PID_S5P" s5p "$PID_WV3" wv3

S2_CKPT="$(find "$OUT_ROOT/s2" -type f -name ckpt_latest.pth | sort | tail -n 1)"
L89_CKPT="$(find "$OUT_ROOT/l89" -type f -name ckpt_best_test.pth | sort | tail -n 1)"
S5P_CKPT="$(find "$OUT_ROOT/s5p" -type f -name ckpt_best_test.pth | sort | tail -n 1)"
WV3_CKPT="$(find "$OUT_ROOT/wv3" -type f -name ckpt_best_test.pth | sort | tail -n 1)"

[ -n "$S2_CKPT" ] && [ -f "$S2_CKPT" ] || { echo "Missing S2_CKPT under $OUT_ROOT/s2" >&2; exit 1; }
[ -n "$L89_CKPT" ] && [ -f "$L89_CKPT" ] || { echo "Missing L89_CKPT under $OUT_ROOT/l89" >&2; exit 1; }
[ -n "$S5P_CKPT" ] && [ -f "$S5P_CKPT" ] || { echo "Missing S5P_CKPT under $OUT_ROOT/s5p" >&2; exit 1; }
[ -n "$WV3_CKPT" ] && [ -f "$WV3_CKPT" ] || { echo "Missing WV3_CKPT under $OUT_ROOT/wv3" >&2; exit 1; }

echo "Using checkpoints:"
echo "  S2  (epoch2 latest): $S2_CKPT"
echo "  L89 (best): $L89_CKPT"
echo "  S5P (best): $S5P_CKPT"
echo "  WV3 (best): $WV3_CKPT"

"$PYTHON_BIN" universal_models_fusion/infer_overlap_or_single_models_native.py \
  --csv_path "$INFER_CSV" --label_column label --device cuda \
  --batch_size 16 --sensor_sub_batch_size 2 --model_resident one_by_one --amp_dtype none --num_workers 8 \
  --s2_ckpt "$S2_CKPT" \
  --l89_ckpt "$L89_CKPT" \
  --s5p_ckpt "$S5P_CKPT" \
  --wv3_ckpt "$WV3_CKPT" \
  --wv3_preprocess_mode train_compat \
  --output_csv "$OUT_ROOT/single4_baselines_infer.csv" \
  --local_cache_dir "$CACHE_DIR" \
  2>&1 | awk 'tolower($0) !~ /warning/ { print; fflush() }' | tee "$OUT_ROOT/logs/infer.log"


echo "DONE"
echo "OUT_ROOT=$OUT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "INFER_OUTPUT=$OUT_ROOT/single4_baselines_infer.csv"
echo "LOG_DIR=$OUT_ROOT/logs"
