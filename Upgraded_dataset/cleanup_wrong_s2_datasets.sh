#!/usr/bin/env bash
set -euo pipefail

mode="${1:---dry-run}"
if [[ "$mode" != "--dry-run" && "$mode" != "--execute" ]]; then
  echo "Usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

wrong_roots=(
  "/mnt/engg-niulab/yuyao/preprocessed_512/S2_6time_historical_bounds_native_v9"
  "/transferdiniu2/yuyao/final_crop/s2_historical_bounds_native_32_v9"
  "/diniuvol/yuyao/s2_v9_vit_cache"
  "/diniuvol/yuyao/s2_point_center_native_dn0_v12_gate_512"
  "/diniuvol/yuyao/s2_v12_gate_cache"
  "/diniuvol/yuyao/s2_v14_crop_cache"
)

protected_roots=(
  "/mnt/engg-niulab/yuyao/preprocessed_512/S2_6time_point_center_plus1000_v14"
  "/mnt/engg-niulab/yuyao/final_crop/s2_6time_legacy_notebook_point_v14_32"
  "/diniuvol/yuyao/s2_6time_legacy_notebook_point_v14_32"
  "/diniuvol/yuyao/s2_6time_point_center_corrected_32"
  "/mnt/engg-niulab/yuyao/sensors_raw_data/S2_GEE_6time"
)

for wrong_root in "${wrong_roots[@]}"; do
  for protected_root in "${protected_roots[@]}"; do
    if [[ "$wrong_root" == "$protected_root" || "$wrong_root" == "$protected_root/"* ]]; then
      echo "Refusing to delete protected path: $wrong_root" >&2
      exit 3
    fi
  done
done

if [[ "$mode" == "--dry-run" ]]; then
  for wrong_root in "${wrong_roots[@]}"; do
    if [[ -e "$wrong_root" ]]; then
      echo "DELETE $wrong_root"
    else
      echo "MISSING $wrong_root"
    fi
  done
  exit 0
fi

pids=()
for wrong_root in "${wrong_roots[@]}"; do
  if [[ -e "$wrong_root" ]]; then
    rm -rf -- "$wrong_root" &
    pids+=("$!")
  fi
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
