#!/usr/bin/env bash
set -Eeuo pipefail

METHANE_PY=/home/yuyao/miniconda3/envs/methane/bin/python
PANOPTICON_PY=/home/yuyao/miniconda3/envs/panopticon/bin/python
PANOPTICON_ROOT=/home/yuyao/panopticon
PIPELINE_ROOT=/home/yuyao/methane_train/Upgrade_data_pipeline

RECROP_CSV="$PIPELINE_ROOT/csv/s2_6time_point_center_exact_v3_paths.csv"
MASK_CSV="$PIPELINE_ROOT/csv/s2_6time_exact_point_center_v3_with_masks.csv"
MASK_QA_CSV="$PIPELINE_ROOT/csv/s2_6time_exact_point_center_v3_mask_qa.csv"
MASK_ROOT=/diniuvol/yuyao/s2_exact_point_masks_v3
SPLIT_ROOT="$PANOPTICON_ROOT/Upgraded_dataset/s2_6time_exact_point_center_v3_temporal_split"
PATCH_ROOT=/diniuvol/yuyao/s2_6time_exact_point_center_v3_32
PATCH_CSV_ROOT="$PANOPTICON_ROOT/Upgraded_dataset/s2_6time_exact_point_center_v3_patches"

AUDIT_SCRIPT="$PANOPTICON_ROOT/Upgraded_dataset/s2_exact_pipeline_audit.py"
MASK_SCRIPT="$PANOPTICON_ROOT/Upgraded_dataset/build_s2_aligned_masks_maxpool.py"
SPLIT_SCRIPT="$PIPELINE_ROOT/code/S2_preprocess/s2_6time_cdse_legacy512_rebuild.py"
CROP_SCRIPT="$PANOPTICON_ROOT/Upgraded_dataset/s2_crop32_to224_local_cache.py"
RESNET_SCRIPT="$PANOPTICON_ROOT/Upgraded_dataset/s2_resnet18_diagnostic.py"

MASTER_ROOT="$PANOPTICON_ROOT/Upgraded_dataset/s2_exact_reports_v3"
RECROP_AUDIT="$MASTER_ROOT/recrop_audit.json"
MASK_AUDIT="$MASTER_ROOT/mask_audit.json"
SPLIT_AUDIT="$MASTER_ROOT/split_audit_verified.json"
PATCH_AUDIT="$MASTER_ROOT/patch_audit.json"
RESNET_JSON="$MASTER_ROOT/resnet18_diagnostic.json"

mkdir -p "$MASTER_ROOT" "$SPLIT_ROOT" "$PATCH_CSV_ROOT" "$MASK_ROOT" "$PATCH_ROOT"
EXPECTED_ROWS=$(
  "$METHANE_PY" -c \
    "import pandas as pd; print(len(pd.read_csv('$RECROP_CSV', low_memory=False)))"
)

stamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(stamp)" "$*"
}

run_stage() {
  local name=$1
  local timeout_seconds=$2
  shift 2
  local started
  started=$(date +%s)
  log "START $name timeout=${timeout_seconds}s"
  timeout --signal=TERM --kill-after=60s "$timeout_seconds" "$@"
  local elapsed=$(( $(date +%s) - started ))
  log "DONE $name elapsed=${elapsed}s"
}

log "Pipeline start"

run_stage recrop_audit 1800 \
  "$METHANE_PY" "$AUDIT_SCRIPT" recrop \
  --csv "$RECROP_CSV" \
  --expected-rows "$EXPECTED_ROWS" \
  --max-center-offset 1.5 \
  --sample-images 3000 \
  --workers 32 \
  --output "$RECROP_AUDIT"

run_stage exact_masks 3600 \
  "$METHANE_PY" "$MASK_SCRIPT" \
  --input-csv "$RECROP_CSV" \
  --gee-local-csv /tmp/no_s2_gee_for_exact_pipeline.csv \
  --out-csv "$MASK_CSV" \
  --qa-csv "$MASK_QA_CSV" \
  --out-root "$MASK_ROOT" \
  --cm-root /mnt/engg-niulab/yuyao/sensors_raw_data/CM \
  --existing-mask-roots /tmp/no_existing_s2_masks \
  --workers 32 \
  --overwrite

run_stage mask_audit 1800 \
  "$METHANE_PY" "$AUDIT_SCRIPT" masks \
  --csv "$MASK_CSV" \
  --expected-rows "$EXPECTED_ROWS" \
  --center-box 256 \
  --workers 32 \
  --output "$MASK_AUDIT"

run_stage temporal_split 600 \
  "$METHANE_PY" "$SPLIT_SCRIPT" split \
  --complete-csv "$MASK_CSV" \
  --split-root "$SPLIT_ROOT" \
  --target-ratio 0.85 \
  --min-ratio 0.80 \
  --max-ratio 0.90

readarray -t SPLIT_PATHS < <(
  "$METHANE_PY" - "$SPLIT_ROOT/split_audit.json" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1]))
print(audit["train_csv"])
print(audit["test_csv"])
PY
)
TRAIN_512_CSV=${SPLIT_PATHS[0]}
TEST_512_CSV=${SPLIT_PATHS[1]}

run_stage split_audit 600 \
  "$METHANE_PY" "$AUDIT_SCRIPT" split \
  --train-csv "$TRAIN_512_CSV" \
  --test-csv "$TEST_512_CSV" \
  --min-ratio 0.80 \
  --max-ratio 0.90 \
  --output "$SPLIT_AUDIT"

run_stage crop32_local 7200 \
  "$METHANE_PY" "$CROP_SCRIPT" \
  --train-csv "$TRAIN_512_CSV" \
  --test-csv "$TEST_512_CSV" \
  --out-root-32 "$PATCH_ROOT" \
  --out-csv-32 "$PATCH_CSV_ROOT/{split}_patches_32.csv" \
  --no-resize-to-224 \
  --source-cache-root /diniuvol/yuyao/s2_512_read_cache_v3 \
  --mask-roots "$MASK_ROOT" \
  --mask-policy require_aligned \
  --label-rule legacy_center_mask_gated \
  --legacy-center-box 10 \
  --n-pos 16 \
  --n-neg 16 \
  --max-tries-pos 800 \
  --max-tries-neg 800 \
  --pos-mask-min-sum 1 \
  --neg-mask-max-sum 0 \
  --neg-require-mask-empty \
  --band-index 11 \
  --zero-ratio-thresh 0.20 \
  --workers 32 \
  --progress-every 100 \
  --flush-every-sec 30 \
  --resume \
  --verbose-failures

TRAIN_PATCH_CSV="$PATCH_CSV_ROOT/train_patches_32.csv"
TEST_PATCH_CSV="$PATCH_CSV_ROOT/test_patches_32.csv"

run_stage patch_audit 1800 \
  "$METHANE_PY" "$AUDIT_SCRIPT" patches \
  --train-csv "$TRAIN_PATCH_CSV" \
  --test-csv "$TEST_PATCH_CSV" \
  --n-pos 16 \
  --n-neg 16 \
  --expected-size 32 \
  --sample-rows 10000 \
  --workers 64 \
  --output "$PATCH_AUDIT"

run_stage resnet18_gate 7200 \
  env CUDA_VISIBLE_DEVICES=0 "$PANOPTICON_PY" "$RESNET_SCRIPT" \
  --train-csv "$TRAIN_PATCH_CSV" \
  --test-csv "$TEST_PATCH_CSV" \
  --mode current \
  --timepoints t0,prev1,prev2,prev3,seasonal,year \
  --band-indices 0,1,2,3,4,5,6,7,10,11 \
  --center-box 0 \
  --max-train-samples 20000 \
  --max-test-samples 10000 \
  --batch-size 512 \
  --epochs 5 \
  --num-workers 16 \
  --device cuda:0 \
  --output "$RESNET_JSON"

"$PANOPTICON_PY" - "$RESNET_JSON" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
history = report["history"]
best_test_auroc = max(float(row["test"]["auroc"]) for row in history)
best_test_f1 = max(float(row["test"]["f1"]) for row in history)
best_train_f1 = max(float(row["train"]["f1"]) for row in history)
summary = {
    "best_train_f1": best_train_f1,
    "best_test_f1": best_test_f1,
    "best_test_auroc": best_test_auroc,
}
print(json.dumps(summary), flush=True)
if best_test_auroc < 0.62:
    raise SystemExit(
        f"ResNet gate failed: best test AUROC {best_test_auroc:.4f} < 0.62"
    )
if best_train_f1 >= 0.85 and best_test_f1 < 0.55:
    raise SystemExit(
        "ResNet gate failed: pathological train/test F1 gap "
        f"{best_train_f1:.4f}/{best_test_f1:.4f}"
    )
PY

log "PIPELINE_READY_FOR_VIT train_csv=$TRAIN_PATCH_CSV test_csv=$TEST_PATCH_CSV"
