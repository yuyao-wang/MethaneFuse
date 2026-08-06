#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yuyao/panopticon
PYTHON=/home/yuyao/miniconda3/envs/methane/bin/python
DN_REPORT="$ROOT/Upgraded_dataset/s2_dn_harmonize_v4_report.json"
LOG="$ROOT/Upgraded_dataset/s2_canonical_recrop_v4.log"
DONE="$ROOT/Upgraded_dataset/s2_canonical_recrop_v4.done"
FAILED="$ROOT/Upgraded_dataset/s2_canonical_recrop_v4.failed"

rm -f "$DONE" "$FAILED"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for DN harmonization"
for _ in $(seq 1 180); do
  if [[ -s "$DN_REPORT" ]]; then
    break
  fi
  sleep 10
done
if [[ ! -s "$DN_REPORT" ]]; then
  echo "DN harmonization report did not appear"
  touch "$FAILED"
  exit 1
fi

"$PYTHON" -c "import json; p='$DN_REPORT'; d=json.load(open(p)); assert not d['failures'], d['failures']; assert d['completed']==d['tasks'], d"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DN harmonization passed"

PRODUCT_ROOTS="/diniuvol/yuyao/s2_cdse_point_repair_cache,/diniuvol/yuyao/s2_early_boundary_products,/mnt/engg-niulab/yuyao/sensors_raw_data/S2/raw_data_dir_s2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_S2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2_90360,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2,/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7"

set +e
timeout --signal=TERM --kill-after=5m 115m \
  "$PYTHON" "$ROOT/Upgraded_dataset/s2_exact_point_recrop.py" \
  --table "$ROOT/Upgraded_dataset/s2_canonical_6510_recrop_tasks_v4.csv" \
  --output-table "$ROOT/Upgraded_dataset/s2_canonical_6510_recrop_progress_v4.csv" \
  --target-root /mnt/engg-niulab/yuyao/sensors_raw_data/S2_canonical_6510_local_safe_recrop_v4 \
  --timepoints t0,prev1,prev2,prev3,seasonal,year \
  --workers 32 \
  --sync-interval 25 \
  --product-scratch-dir /diniuvol/yuyao/s2_canonical_product_cache_v4 \
  --crop-scratch-dir /diniuvol/yuyao/s2_canonical_crop_cache_v4 \
  --existing-product-roots "$PRODUCT_ROOTS" \
  --existing-cache-mode adaptive \
  --full-stage-min-tasks 6 \
  --missing-source aws
status=$?
set -e
if [[ "$status" -ne 0 ]]; then
  echo "recrop exited status=$status"
  touch "$FAILED"
  exit "$status"
fi

"$PYTHON" "$ROOT/Upgraded_dataset/s2_build_canonical_6510_manifest.py"
"$PYTHON" -c "import pandas as pd; p='$ROOT/Upgraded_dataset/s2_canonical_6510_complete_v4.csv'; d=pd.read_csv(p); assert len(d)>=5200, len(d); print('complete_rows',len(d))"

printf 'done\n' > "$DONE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] canonical recrop complete"
