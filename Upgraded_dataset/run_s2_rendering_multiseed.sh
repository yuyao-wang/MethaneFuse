#!/usr/bin/env bash
#
# Seed replicates of the two S2 rendering arms (legacy 224 vs new GEE) on the
# event-disjoint split. The original campaign ran one seed per arm, so the
# headline 0.9640 vs 0.8389 AUROC gap has no seed variance attached. Recipe is
# byte-identical to the original arms (batch 32, max 10 / min 8, patience 2,
# noam warmup 4000, full finetune); the only change is --seed.
#
# Usage: run_s2_rendering_multiseed.sh <legacy|gee> <seed>

set -euo pipefail

side="${1:?usage: $0 <legacy|gee> <seed>}"
seed="${2:?usage: $0 <legacy|gee> <seed>}"

root="/home/yuyao/panopticon/Upgraded_dataset"
splits="${root}/s2_legacy360_matched_gee_splits"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"

case "${side}" in
  legacy) split_root="${splits}/legacy_same_rows_splits" ;;
  gee)    split_root="${splits}/gee_same_rows_splits" ;;
  *) echo "side must be legacy or gee" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=0
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1

exec "${python_bin}" "${root}/run_s2_convergence_queue.py" \
  --side "${side}" \
  --split-root "${split_root}" \
  --checkpoint-root "/diniuvol/yuyao/checkpoints/s2_rendering_multiseed_20260731/${side}_seed${seed}" \
  --result-root "${splits}/multiseed_20260731/${side}_seed${seed}" \
  --run-prefix "s2_rendering_${side}_seed${seed}" \
  --python "${python_bin}" \
  --train-script "${root}/dino_classifier_head_s2_legacy360_repro.py" \
  --weights "/home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth" \
  --max-epochs 10 \
  --min-epoch 8 \
  --patience 2 \
  --min-delta 0.001 \
  --early-stop-metric test_auroc \
  --batch-size 32 \
  --num-workers 4 \
  --seed "${seed}" \
  --splits event_disjoint_80_20
