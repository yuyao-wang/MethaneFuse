#!/usr/bin/env bash
#
# Forecasting probe: predict whether the current visit carries a plume using
# ONLY earlier visits. t0 is withheld from the model entirely (temporal_mode
# history -> 24 channels from t90/t360; no zero padding, no t0 channel ids).
#
# Run on the event-disjoint split so the question is "does an unseen site's
# history predict its current emission state", not "have we memorised this
# event".
#
# Usage: run_s2_history_only_forecast.sh <legacy|gee> <gpu_index> <mode>
#   mode: history          two earlier visits (the forecasting arm)
#         history_oldest   single earliest visit (site-identity control)

set -euo pipefail

side="${1:?usage: $0 <legacy|gee> <gpu_index> <mode>}"
gpu="${2:?}"
mode="${3:-history}"

root="/home/yuyao/panopticon/Upgraded_dataset"
splits="${root}/s2_legacy360_matched_gee_splits"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"

case "${side}" in
  legacy) split_root="${splits}/legacy_same_rows_splits" ;;
  gee)    split_root="${splits}/gee_same_rows_splits" ;;
  *) echo "unknown side ${side}" >&2; exit 2 ;;
esac

tag="${side}_${mode}"
out="${splits}/history_forecast_20260730/${tag}"
ckpt="/transferdiniu2/yuyao/checkpoints/s2_history_forecast_20260730/${tag}"
mkdir -p "${out}" "${ckpt}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1

# Short probe: we only need to know whether signal exists above chance, so this
# runs a fixed 5 epochs rather than the 8-10 epoch convergence protocol used for
# the detection arms. Numbers are therefore not comparable to those runs as
# final results.
for epoch in 1 2 3 4 5; do
  resume=()
  [ "${epoch}" -gt 1 ] && resume=(--resume)
  "${python_bin}" "${root}/dino_classifier_head_s2_legacy360_repro.py" \
    --train_csv "${split_root}/event_disjoint_80_20/train.csv" \
    --test_csv "${split_root}/event_disjoint_80_20/test.csv" \
    --weights "/home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth" \
    --batch_size 32 --epochs "${epoch}" \
    --head_lr 0.001 --backbone_lr 0.0001 \
    --lr_scheduler noam --warmup_steps 4000 \
    --weight_decay 0.0005 --momentum 0.9 \
    --num_workers 4 --pad_to_multiple 1 --input_resize_size 224 \
    --temporal_mode "${mode}" \
    --device cuda:0 --log_interval 200 --seed 20260730 \
    --t0_col s2_0_path --t90_col s2_90_path --t360_col s2_360_path \
    --train_backbone --freeze_backbone_epochs 0 --max_grad_norm 1.0 \
    --save_checkpoints --checkpoint_dir "${ckpt}" \
    --run_name "${tag}" \
    --metrics_jsonl "${out}/metrics.jsonl" "${resume[@]}"
done

echo "[done] ${tag}"
