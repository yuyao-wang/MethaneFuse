#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/yuyao/panopticon
PY=/home/yuyao/miniconda3/envs/panopticon/bin/python
PATCH_CSV_ROOT="$ROOT/Upgraded_dataset/s2_6time_exact_point_center_v3_patches"
GATE_CSV_ROOT="$ROOT/Upgraded_dataset/s2_6time_exact_point_center_v3_vit_gate"
CHECKPOINT_ROOT=/transferdiniu2/yuyao/checkpoints/s2_exact_point_center_v3
RUN_NAME=s2_6time_exact_point_center_v3_vit_gate_full_finetune
LOG="$ROOT/Upgraded_dataset/s2_exact_reports_v3/vit_gate.log"

mkdir -p "$GATE_CSV_ROOT" "$ROOT/Upgraded_dataset/s2_exact_reports_v3"

"$PY" "$ROOT/Upgraded_dataset/s2_make_balanced_gate_subset.py" \
  --train-csv "$PATCH_CSV_ROOT/train_patches_32.csv" \
  --test-csv "$PATCH_CSV_ROOT/test_patches_32.csv" \
  --out-root "$GATE_CSV_ROOT" \
  --max-train-rows 16000 \
  --max-test-rows 8000 \
  --seed 20260723

cd "$ROOT"
timeout --signal=TERM --kill-after=120s 14400 \
  env CUDA_VISIBLE_DEVICES=0 "$PY" \
  "$ROOT/Upgraded_dataset/dino_classifier_head_s2_temporal_satmae.py" \
  --train_csv "$GATE_CSV_ROOT/train_gate.csv" \
  --test_csv "$GATE_CSV_ROOT/test_gate.csv" \
  --weights "$ROOT/weights/panopticon_vitb14_teacher.pth" \
  --train_backbone \
  --channel_indices 0,1,2,3,4,5,6,7,10,11 \
  --input_resize_size 224 \
  --batch_size 16 \
  --epochs 3 \
  --head_lr 1e-3 \
  --temporal_lr 1e-3 \
  --backbone_lr 1e-4 \
  --lr_scheduler none \
  --num_workers 16 \
  --prefetch_factor 2 \
  --stats_samples 1000 \
  --stats_workers 16 \
  --local_cache_mode off \
  --local_cache_dir "" \
  --checkpoint_dir "$CHECKPOINT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda:0 \
  --num_gpus 1 \
  --log_interval 50 \
  2>&1 | tee "$LOG"

"$PY" - "$LOG" <<'PY'
import json
import re
import sys

pattern = re.compile(
    r"Epoch (?P<epoch>\d+): .*?"
    r"train_f1=(?P<train_f1>[0-9.]+) .*?"
    r"test_f1=(?P<test_f1>[0-9.]+) .*?"
    r"auroc=(?P<auroc>[0-9.]+)"
)
records = [
    {
        "epoch": int(match.group("epoch")),
        "train_f1": float(match.group("train_f1")),
        "test_f1": float(match.group("test_f1")),
        "test_auroc": float(match.group("auroc")),
    }
    for match in pattern.finditer(open(sys.argv[1]).read())
]
print(json.dumps(records, indent=2), flush=True)
if len(records) != 3:
    raise SystemExit(f"ViT gate produced {len(records)}/3 epoch summaries")
best = {
    "train_f1": max(row["train_f1"] for row in records),
    "test_f1": max(row["test_f1"] for row in records),
    "test_auroc": max(row["test_auroc"] for row in records),
}
print(json.dumps({"best": best}), flush=True)
if best["test_auroc"] < 0.60:
    raise SystemExit(
        f"ViT gate failed: best test AUROC {best['test_auroc']:.4f} < 0.60"
    )
if best["train_f1"] >= 0.80 and best["test_f1"] < 0.55:
    raise SystemExit(
        "ViT gate failed: pathological train/test F1 gap "
        f"{best['train_f1']:.4f}/{best['test_f1']:.4f}"
    )
PY

printf 'VIT_GATE_PASSED\n'
