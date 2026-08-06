#!/usr/bin/env bash
set -euo pipefail

# Historical engineering protocol only.  The universal warm-start checkpoint
# was selected with feedback from the old evaluation split, so this launcher is
# for rapid architecture iteration and must not be reported as leakage-free.

repo_root=${REPO_ROOT:-/home/yuyao/panopticon}
python_bin=${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}
train_csv=${TRAIN_CSV:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_360m/manifest_time_train.csv}
dev_csv=${DEV_CSV:-/diniuvol/yuyao/methanefuse_research_20260727/legacy360_patch_axial/manifests/development.csv}
test_csv=${TEST_CSV:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_360m/manifest_time_test.csv}
checkpoint=${CHECKPOINT:-/transferdiniu2/yuyao/checkpoints/universal_360m/query dataset 360m/ckpt_best_test.pth}
output_dir=${OUTPUT_DIR:-/diniuvol/yuyao/methanefuse_research_20260727/legacy360_patch_axial/universal_seed41}
raw_cache=${RAW_CACHE:-/diniuvol/yuyao/methanefuse_research_20260727/legacy360_patch_axial/raw_file_cache}
device=${DEVICE:-cuda:0}

for required in "$train_csv" "$dev_csv" "$test_csv" "$checkpoint"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

extra_args=()
if [[ -n ${MAX_TRAIN_STEPS:-} ]]; then
  extra_args+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi

cd "$repo_root"
exec "$python_bin" research/pretraining_20260727/legacy360_patch_axial.py \
  --train_csv "$train_csv" \
  --dev_csv "$dev_csv" \
  --test_csv "$test_csv" \
  --sealed_test_authorization engineering-historical-final-eval \
  --checkpoint "$checkpoint" \
  --output_dir "$output_dir" \
  --raw_cache_dir "$raw_cache" \
  --warm_cache \
  --device "$device" \
  --epochs 4 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --workers 12 \
  --cache_workers 32 \
  --freeze_backbone_epochs 1 \
  --backbone_lr 1e-5 \
  --residual_lr 3e-4 \
  --temporal_frame_blocks 2 \
  --topk_fraction 0.25 \
  --sensor_aux_weight 0.2 \
  --selection_metric binary_f1 \
  "${extra_args[@]}"
