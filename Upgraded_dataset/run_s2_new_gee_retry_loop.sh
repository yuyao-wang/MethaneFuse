#!/usr/bin/env bash
set -u

PYTHON=/home/yuyao/miniconda3/envs/methane/bin/python
ROOT=/home/yuyao/methane_train
EXPORTER="$ROOT/Upgrade_data_pipeline/code/S2_preprocess/s2_6time_gee_export.py"
INPUT="$ROOT/Upgrade_data_pipeline/csv/s2_gee_retry_inputs/s2_gee_retry_any_failed_plume.csv"
MANIFEST="$ROOT/Upgrade_data_pipeline/csv/s2_6time_gee_export_retry.csv"
POLL_SECONDS=${POLL_SECONDS:-600}

while pgrep -f '[s]2_6time_gee_export.py.*s2_gee_retry_any_failed_plume.csv' >/dev/null; do
  printf '[new-export-loop] existing retry exporter is still running; checking again in %ss\n' "$POLL_SECONDS"
  sleep "$POLL_SECONDS"
done

while true; do
  "$PYTHON" "$EXPORTER" \
    --input-csv "$INPUT" \
    --out-manifest "$MANIFEST" \
    --timepoints all \
    --workers 2 \
    --max-active-tasks 500 \
    --max-drive-files 500 \
    --resume
  status=$?
  printf '[new-export-loop] exporter exit=%s; retrying in %ss\n' "$status" "$POLL_SECONDS"
  sleep "$POLL_SECONDS"
done
