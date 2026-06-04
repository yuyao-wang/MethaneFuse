#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-0}
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}

TMP_BASE=${TMP_BASE:-/transferdiniu2/yuyao/temp}
mkdir -p "$TMP_BASE"
chmod 700 "$TMP_BASE"
export TMPDIR="$TMP_BASE"
export TMP="$TMP_BASE"
export TEMP="$TMP_BASE"

DATA_ROOT=${DATA_ROOT:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m_518}
TRAIN_CSV=${TRAIN_CSV:-$DATA_ROOT/manifest_time_train.csv}
TEST_CSV=${TEST_CSV:-$DATA_ROOT/manifest_time_test.csv}
RUN_LABEL=${RUN_LABEL:-query_dataset_480m_518}
PYTHON_BIN=${PYTHON_BIN:-python}
WEIGHTS=${WEIGHTS:-weights/panopticon_vitb14_teacher.pth}
CACHE_DIR=${CACHE_DIR:-/diniuvol/yuyao/local_train_temp_cache_480m_518}
OUT_ROOT=${OUT_ROOT:-/transferdiniu2/yuyao/checkpoints}
QUERY_MODE=${QUERY_MODE:-all}

PRETRAIN_CHECKPOINT_DIR=${PRETRAIN_CHECKPOINT_DIR:-$OUT_ROOT/methanefuse_pretrain}
SEG_CHECKPOINT_DIR=${SEG_CHECKPOINT_DIR:-$OUT_ROOT/methanefuse_segmentation}
SEG_BACKBONE_PATH=${SEG_BACKBONE_PATH:-$TMPDIR/query_backbone_for_seg.pth}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-$PRETRAIN_CHECKPOINT_DIR/$RUN_LABEL/ckpt_best_test.pth}

[ -f "$TRAIN_CSV" ] || { echo "Missing TRAIN_CSV: $TRAIN_CSV" >&2; exit 1; }
[ -f "$TEST_CSV" ] || { echo "Missing TEST_CSV: $TEST_CSV" >&2; exit 1; }
[ -f "$WEIGHTS" ] || { echo "Missing WEIGHTS: $WEIGHTS" >&2; exit 1; }
mkdir -p "$PRETRAIN_CHECKPOINT_DIR" "$SEG_CHECKPOINT_DIR"

if [[ "$QUERY_MODE" == "pretrain" || "$QUERY_MODE" == "all" ]]; then
  "$PYTHON_BIN" src/models/pretrain_multisensor.py \
    --train_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV" \
    --weights "$WEIGHTS" \
    --batch_size 12 \
    --epochs 7 \
    --train_backbone \
    --freeze_backbone_epochs 1 \
    --backbone_lr 5e-5 \
    --head_lr 1e-3 \
    --sensor_aux_loss_weight 0.3 \
    --num_workers 8 \
    --device cuda \
    --checkpoint_dir "$PRETRAIN_CHECKPOINT_DIR" \
    --use_wandb \
    --wandb_project query_dataset \
    --wandb_run_name "$RUN_LABEL" \
    --local_cache_dir "$CACHE_DIR" \
    --local_cache_min_free_gb 100 \
    --row_fusion_mode max \
    --local_cache_warmup \
    --local_cache_workers 16
fi

if [[ "$QUERY_MODE" == "seg" || "$QUERY_MODE" == "all" ]]; then
  [ -f "$PRETRAIN_CKPT" ] || {
    echo "Missing PRETRAIN_CKPT: $PRETRAIN_CKPT" >&2
    echo "Set PRETRAIN_CKPT explicitly or run QUERY_MODE=pretrain first for RUN_LABEL='$RUN_LABEL'." >&2
    exit 1
  }

  PRETRAIN_CKPT="$PRETRAIN_CKPT" SEG_BACKBONE_PATH="$SEG_BACKBONE_PATH" "$PYTHON_BIN" - <<'PY'
import os
import torch

src = os.environ["PRETRAIN_CKPT"]
dst = os.environ["SEG_BACKBONE_PATH"]
ckpt = torch.load(src, map_location="cpu")
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
if any(k.startswith("module.") for k in state):
    state = {k[7:]: v for k, v in state.items()}
backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
torch.save({"backbone": backbone}, dst)
print("saved:", dst, "num_backbone_keys:", len(backbone))
PY

  "$PYTHON_BIN" src/models/segmentation.py \
    --train_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV" \
    --tasks s2,l89,emit \
    --weights "$SEG_BACKBONE_PATH" \
    --checkpoint_dir "$SEG_CHECKPOINT_DIR" \
    --wandb_project query_dataset \
    --wandb_run_name "${RUN_LABEL}_seg" \
    --batch_size 12 \
    --epochs 7 \
    --freeze_backbone_epochs 2 \
    --backbone_lr 5e-5 \
    --head_lr 1e-3 \
    --local_cache_dir "$CACHE_DIR" \
    --local_cache_min_free_gb 100
fi
