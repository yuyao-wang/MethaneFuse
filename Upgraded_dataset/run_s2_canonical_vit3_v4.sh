#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon
PYTHON=/home/yuyao/miniconda3/envs/panopticon/bin/python
GATE_DONE="$ROOT/Upgraded_dataset/s2_canonical_crop_gate_v4.done"
GATE_FAILED="$ROOT/Upgraded_dataset/s2_canonical_signal_v4.failed"
LOG="$ROOT/Upgraded_dataset/s2_canonical_vit3_v4.log"
DONE="$ROOT/Upgraded_dataset/s2_canonical_vit3_v4.done"
FAILED="$ROOT/Upgraded_dataset/s2_canonical_vit3_v4.failed"
PATCH_ROOT=/transferdiniu2/yuyao/manifests/s2_canonical_legacy_center_patches_v5
CACHE_ROOT=/diniuvol/yuyao/s2_canonical_train_cache_v4

rm -f "$DONE" "$FAILED"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ResNet signal gate"
for _ in $(seq 1 3240); do
  if [[ -e "$GATE_FAILED" ]]; then
    echo "ResNet signal gate failed; ViT remains stopped"
    touch "$FAILED"
    exit 3
  fi
  if [[ -e "$GATE_DONE" ]]; then
    break
  fi
  sleep 10
done
if [[ ! -e "$GATE_DONE" ]]; then
  echo "ResNet signal gate did not finish"
  touch "$FAILED"
  exit 1
fi

rm -rf "$CACHE_ROOT"
mkdir -p "$CACHE_ROOT"

timeout --signal=TERM --kill-after=10m 8h \
  "$PYTHON" "$ROOT/Upgraded_dataset/dino_classifier_head_s2_temporal_satmae.py" \
  --train_csv "$PATCH_ROOT/train_patches_32.csv" \
  --test_csv "$PATCH_ROOT/test_patches_32.csv" \
  --weights "$ROOT/weights/panopticon_vitb14_teacher.pth" \
  --train_backbone \
  --channel_indices 0,1,2,3,4,5,6,7,10,11 \
  --input_resize_size 224 \
  --batch_size 16 \
  --epochs 3 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --stats_samples 2000 \
  --stats_workers 16 \
  --local_cache_mode sync \
  --local_cache_dir "$CACHE_ROOT" \
  --local_cache_warmup \
  --local_cache_workers 32 \
  --checkpoint_dir /transferdiniu2/yuyao/checkpoints/s2_canonical_legacy_center_v4 \
  --run_name s2_canonical_legacy_center_v4_full_finetune_vit3 \
  --log_interval 100

printf 'done\n' > "$DONE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ViT three-epoch validation complete"
