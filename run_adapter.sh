#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate panopticon
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

TMP_ROOT="${TMP_ROOT:-/transferdiniu2/yuyao/temp}"
mkdir -p "$TMP_ROOT"
chmod 700 "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
export TMP="$TMP_ROOT"
export TEMP="$TMP_ROOT"

DATA_ROOT=/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m
TRAIN_CSV="$DATA_ROOT/manifest_time_train.csv"
TEST_CSV="$DATA_ROOT/manifest_time_test.csv"
WEIGHTS="${WEIGHTS:-weights/panopticon_vitb14_teacher.pth}"
CACHE_DIR="${CACHE_DIR:-/diniuvol/yuyao/local_train_temp_cache_480m}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/transferdiniu2/yuyao/checkpoints/multi_sensor_moe_adapter}"
RUN_NAME="${RUN_NAME:-moe_postblock_stageA_vit_stageB_adapter}"

[ -f "$TRAIN_CSV" ] || { echo "Missing TRAIN_CSV: $TRAIN_CSV" >&2; exit 1; }
[ -f "$TEST_CSV" ] || { echo "Missing TEST_CSV: $TEST_CSV" >&2; exit 1; }
[ -f "$WEIGHTS" ] || { echo "Missing WEIGHTS: $WEIGHTS" >&2; exit 1; }

python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_moeadapter.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --weights "$WEIGHTS" \
  --device cuda \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --local_cache_dir "$CACHE_DIR" \
  --local_cache_min_free_gb 200 \
  --batch_size 24 \
  --epochs 50 \
  --num_workers 8 \
  --head_lr 1e-3 \
  --backbone_lr 5e-5 \
  --adapter_lr 1e-4 \
  --training_schedule adapter_then_joint \
  --stage_a_adapter_epochs 3 \
  --stage_b_adapter_lr 1e-4 \
  --stage_b_backbone_to_adapter_lr_ratio 0.4 \
  --adapter_bottleneck_dim 16 \
  --adapter_alpha 16 \
  --adapter_dropout 0.05 \
  --adapter_token_blocks 6 \
  --sensor_aux_loss_weight 0.4 \
  --sensor_aux_effective_num_beta 0.999 \
  --consistency_loss_weight 0.05 \
  --consistency_temperature 1.0 \
  --overlap_sensor_weights "s2=1.1,l89=1.0,wv3=1.0,s5p=0.6" \
  --overlap_loss_weight 0.15 \
  --sensor_stats_cache "$CHECKPOINT_DIR/sensor_stats_moe_adapter.json" \
  --sensor_stats_seed 42 \
  --use_wandb \
  --wandb_project query_dataset \
  --wandb_run_name "$RUN_NAME"

LORA_CHECKPOINT_DIR="${LORA_CHECKPOINT_DIR:-/transferdiniu2/yuyao/checkpoints/multi_sensor_qv_lora_adapter}"
LORA_RUN_NAME="${LORA_RUN_NAME:-qv_lora_stageA3_frozen_dino_stageB}"

python universal_models_fusion/multi_sensor_panopticon_4_lora_adapter.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --weights "$WEIGHTS" \
  --device cuda \
  --checkpoint_dir "$LORA_CHECKPOINT_DIR" \
  --local_cache_dir "$CACHE_DIR" \
  --local_cache_min_free_gb 200 \
  --batch_size 24 \
  --epochs 50 \
  --num_workers 8 \
  --head_lr 1e-3 \
  --backbone_lr 5e-5 \
  --adapter_lr 1e-4 \
  --stage_a_epochs 3 \
  --adapter_blocks 12 \
  --adapter_bottleneck_dim 16 \
  --adapter_alpha 16 \
  --adapter_dropout 0.05 \
  --train_backbone \
  --sensor_aux_loss_weight 0.4 \
  --use_wandb \
  --wandb_project query_dataset \
  --wandb_run_name "$LORA_RUN_NAME"

LORAMOE_CHECKPOINT_DIR="${LORAMOE_CHECKPOINT_DIR:-/transferdiniu2/yuyao/checkpoints/multi_sensor_qv_loramoe_adapter}"
LORAMOE_STAGE_A_RUN_NAME="${LORAMOE_STAGE_A_RUN_NAME:-qv_loramoe_stageA_original}"
LORAMOE_STAGE_B_RUN_NAME="${LORAMOE_STAGE_B_RUN_NAME:-qv_loramoe_stageB_frozen_backbone}"
LORAMOE_STAGE_A_CKPT="${LORAMOE_STAGE_A_CKPT:-$LORAMOE_CHECKPOINT_DIR/$LORAMOE_STAGE_A_RUN_NAME/ckpt_best_test.pth}"

python universal_models_fusion/multi_sensor_panopticon_4_loramoe_adapter.py \
  --stage a \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --weights "$WEIGHTS" \
  --device cuda \
  --checkpoint_dir "$LORAMOE_CHECKPOINT_DIR" \
  --local_cache_dir "$CACHE_DIR" \
  --local_cache_min_free_gb 200 \
  --batch_size 24 \
  --epochs 50 \
  --num_workers 8 \
  --head_lr 1e-3 \
  --backbone_lr 5e-5 \
  --train_backbone \
  --sensor_aux_loss_weight 0.4 \
  --use_wandb \
  --wandb_project query_dataset \
  --wandb_run_name "$LORAMOE_STAGE_A_RUN_NAME"

[ -f "$LORAMOE_STAGE_A_CKPT" ] || { echo "Missing LORAMOE_STAGE_A_CKPT: $LORAMOE_STAGE_A_CKPT" >&2; exit 1; }

python universal_models_fusion/multi_sensor_panopticon_4_loramoe_adapter.py \
  --stage b \
  --stage_a_checkpoint "$LORAMOE_STAGE_A_CKPT" \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --weights "$WEIGHTS" \
  --device cuda \
  --checkpoint_dir "$LORAMOE_CHECKPOINT_DIR" \
  --local_cache_dir "$CACHE_DIR" \
  --local_cache_min_free_gb 200 \
  --batch_size 24 \
  --epochs 50 \
  --num_workers 8 \
  --head_lr 1e-3 \
  --backbone_lr 1e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --sensor_aux_loss_weight 0.4 \
  --use_wandb \
  --wandb_project query_dataset \
  --wandb_run_name "$LORAMOE_STAGE_B_RUN_NAME"
