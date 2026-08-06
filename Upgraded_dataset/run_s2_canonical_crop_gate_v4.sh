#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon
PY_METHANE=/home/yuyao/miniconda3/envs/methane/bin/python
PY_PANOPTICON=/home/yuyao/miniconda3/envs/panopticon/bin/python
RETRY_DONE="$ROOT/Upgraded_dataset/s2_canonical_local_retry_v4.done"
LOG="$ROOT/Upgraded_dataset/s2_canonical_crop_gate_v4.log"
DONE="$ROOT/Upgraded_dataset/s2_canonical_crop_gate_v4.done"
FAILED="$ROOT/Upgraded_dataset/s2_canonical_crop_gate_v4.failed"
SIGNAL_FAILED="$ROOT/Upgraded_dataset/s2_canonical_signal_v4.failed"

MASKED_CSV="$ROOT/Upgraded_dataset/s2_canonical_6510_complete_with_masks_v4.csv"
MASK_QA="$ROOT/Upgraded_dataset/s2_canonical_masks_qa_v4.csv"
MASK_ROOT=/diniuvol/yuyao/s2_canonical_masks_v4
SPLIT_ROOT="$ROOT/Upgraded_dataset/s2_canonical_temporal_split_v4"
PATCH_ROOT=/transferdiniu2/yuyao/final_crop/s2_canonical_legacy_center_32_v5
PATCH_CSV_ROOT=/transferdiniu2/yuyao/manifests/s2_canonical_legacy_center_patches_v5
RESNET_REPORT="$ROOT/Upgraded_dataset/s2_canonical_resnet_gate_v4.json"
RESNET_LEGACY3_REPORT="$ROOT/Upgraded_dataset/s2_canonical_resnet_gate_legacy3_v4.json"
RESUME_CROP="${S2_CROP_RESUME:-0}"
SKIP_CROP="${S2_SKIP_CROP:-0}"

rm -f "$DONE" "$FAILED" "$SIGNAL_FAILED"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for local tile retry"
for _ in $(seq 1 900); do
  if [[ -e "$RETRY_DONE" ]]; then
    break
  fi
  sleep 10
done
if [[ ! -e "$RETRY_DONE" ]]; then
  echo "local tile retry did not finish"
  touch "$FAILED"
  exit 1
fi

if [[ -s "$MASKED_CSV" && -s "$MASK_QA" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] reusing aligned masks"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] building aligned masks"
  "$PY_METHANE" "$ROOT/Upgraded_dataset/build_s2_aligned_masks_maxpool.py" \
    --input-csv "$ROOT/Upgraded_dataset/s2_canonical_6510_complete_v4.csv" \
    --gee-local-csv /tmp/no_s2_gee_paths_v4.csv \
    --out-csv "$MASKED_CSV" \
    --qa-csv "$MASK_QA" \
    --out-root "$MASK_ROOT" \
    --cm-root /mnt/engg-niulab/yuyao/sensors_raw_data/CM \
    --existing-mask-roots \
      /diniuvol/yuyao/s2_exact_point_masks_v3 \
      /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_raw_s2_90360 \
    --workers 32
fi

if [[ -s "$SPLIT_ROOT/split_audit.json" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] reusing temporal split"
else
  rm -rf "$SPLIT_ROOT"
  "$PY_METHANE" "$ROOT/Upgraded_dataset/s2_6time_cdse_legacy512_rebuild.py" split \
    --complete-csv "$MASKED_CSV" \
    --split-root "$SPLIT_ROOT" \
    --target-ratio 0.85 \
    --min-ratio 0.80 \
    --max-ratio 0.90
fi

TRAIN_CSV=$("$PY_METHANE" -c "import json; print(json.load(open('$SPLIT_ROOT/split_audit.json'))['train_csv'])")
TEST_CSV=$("$PY_METHANE" -c "import json; print(json.load(open('$SPLIT_ROOT/split_audit.json'))['test_csv'])")
"$PY_METHANE" "$ROOT/Upgraded_dataset/s2_exact_pipeline_audit.py" split \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --min-ratio 0.80 \
  --max-ratio 0.90 \
  --output "$SPLIT_ROOT/split_independent_audit.json"

if [[ "$SKIP_CROP" == "1" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] using completed transfer crop"
else
  if [[ "$RESUME_CROP" == "1" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] resuming existing transfer crop"
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] deleting wrong derived patch data"
    rm -rf /diniuvol/yuyao/s2_6time_exact_point_center_v3_32
    rm -rf "$PATCH_ROOT"
    rm -rf "$PATCH_CSV_ROOT"
  fi
  rm -rf /diniuvol/yuyao/s2_canonical_512_crop_cache_v4
  rm -rf /diniuvol/yuyao/s2_canonical_32_output_cache_v4
  mkdir -p "$PATCH_CSV_ROOT"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cropping exact legacy-center 32 patches"
  timeout --signal=TERM --kill-after=5m 115m \
    "$PY_PANOPTICON" "$ROOT/Upgraded_dataset/s2_crop32_to224_local_cache.py" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    --out-root-32 "$PATCH_ROOT" \
    --out-csv-32 "$PATCH_CSV_ROOT/{split}_patches_32.csv" \
    --no-resize-to-224 \
    --label-rule legacy_center_anchor \
    --mask-policy legacy_zero_if_missing \
    --no-forbid-mask512-fallback \
    --mask-roots "$MASK_ROOT" \
    --legacy-positive-center-size 20 \
    --legacy-center-box 10 \
    --quality-timepoints t0,seasonal,year \
    --n-pos 16 \
    --n-neg 16 \
    --workers 32 \
    --source-cache-root /diniuvol/yuyao/s2_canonical_512_crop_cache_v4 \
    --output-cache-root /diniuvol/yuyao/s2_canonical_32_output_cache_v4 \
    --progress-every 200 \
    --flush-every-sec 120 \
    --seed 20260723
fi

PATCH_TRAIN="$PATCH_CSV_ROOT/train_patches_32.csv"
PATCH_TEST="$PATCH_CSV_ROOT/test_patches_32.csv"
"$PY_METHANE" "$ROOT/Upgraded_dataset/s2_exact_pipeline_audit.py" patches \
  --train-csv "$PATCH_TRAIN" \
  --test-csv "$PATCH_TEST" \
  --n-pos 16 \
  --n-neg 16 \
  --allow-partial-plumes \
  --notebook-cell7-geometry \
  --expected-size 32 \
  --sample-rows 10000 \
  --workers 32 \
  --no-check-mask-label-consistency \
  --no-check-all-valid-bands \
  --quality-path-columns path_t0,path_seasonal,path_year \
  --quality-band-index 11 \
  --quality-zero-ratio-thresh 0.20 \
  --output "$PATCH_CSV_ROOT/patch_audit_v4.json"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running legacy-three-timepoint ResNet control"
"$PY_PANOPTICON" "$ROOT/Upgraded_dataset/s2_resnet18_diagnostic.py" \
  --train-csv "$PATCH_TRAIN" \
  --test-csv "$PATCH_TEST" \
  --mode current \
  --timepoints t0,seasonal,year \
  --band-indices 0,1,2,3,4,5,6,7,10,11 \
  --center-box 0 \
  --max-train-samples 30000 \
  --max-test-samples 12000 \
  --batch-size 512 \
  --epochs 3 \
  --num-workers 16 \
  --device cuda:0 \
  --output "$RESNET_LEGACY3_REPORT"

set +e
"$PY_PANOPTICON" -c "import json; d=json.load(open('$RESNET_LEGACY3_REPORT')); h=d['history']; best_auc=max(x['test']['auroc'] for x in h[-2:]); best_f1=max(x['test']['f1'] for x in h[-2:]); gap_auc=h[-1]['train']['auroc']-h[-1]['test']['auroc']; gap_f1=h[-1]['train']['f1']-h[-1]['test']['f1']; print('legacy3 best_test_auroc',best_auc,'best_test_f1',best_f1,'final_auc_gap',gap_auc,'final_f1_gap',gap_f1); assert best_auc>=0.62 and best_f1>=0.60 and gap_auc<=0.18 and gap_f1<=0.20"
legacy_gate_status=$?
set -e
if [[ "$legacy_gate_status" -ne 0 ]]; then
  echo "legacy-three-timepoint signal gate failed; six-time and ViT will not start"
  printf 'legacy3_failed\n' > "$SIGNAL_FAILED"
  exit 3
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running all-six-timepoint ResNet signal gate"
"$PY_PANOPTICON" "$ROOT/Upgraded_dataset/s2_resnet18_diagnostic.py" \
  --train-csv "$PATCH_TRAIN" \
  --test-csv "$PATCH_TEST" \
  --mode current \
  --timepoints t0,prev1,prev2,prev3,seasonal,year \
  --band-indices 0,1,2,3,4,5,6,7,10,11 \
  --center-box 0 \
  --max-train-samples 30000 \
  --max-test-samples 12000 \
  --batch-size 512 \
  --epochs 3 \
  --num-workers 16 \
  --device cuda:0 \
  --output "$RESNET_REPORT"

set +e
"$PY_PANOPTICON" -c "import json; d=json.load(open('$RESNET_REPORT')); h=d['history']; best_auc=max(x['test']['auroc'] for x in h[-2:]); best_f1=max(x['test']['f1'] for x in h[-2:]); gap_auc=h[-1]['train']['auroc']-h[-1]['test']['auroc']; gap_f1=h[-1]['train']['f1']-h[-1]['test']['f1']; print('best_test_auroc',best_auc,'best_test_f1',best_f1,'final_auc_gap',gap_auc,'final_f1_gap',gap_f1); assert best_auc>=0.62 and best_f1>=0.60 and gap_auc<=0.18 and gap_f1<=0.20"
gate_status=$?
set -e
if [[ "$gate_status" -ne 0 ]]; then
  echo "signal gate failed; ViT will not start"
  printf 'failed\n' > "$SIGNAL_FAILED"
  exit 3
fi

rm -rf /diniuvol/yuyao/s2_canonical_512_crop_cache_v4
rm -rf /diniuvol/yuyao/s2_canonical_32_output_cache_v4
printf 'done\n' > "$DONE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] crop and signal gate passed"
