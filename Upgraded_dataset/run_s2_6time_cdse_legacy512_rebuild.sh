#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yuyao/miniconda3/envs/panopticon/bin/python}"
PANOPTICON_ROOT="${PANOPTICON_ROOT:-/home/yuyao/panopticon}"
PIPELINE="${PIPELINE:-${PANOPTICON_ROOT}/Upgraded_dataset/s2_6time_cdse_legacy512_rebuild.py}"

CSV_ROOT="${CSV_ROOT:-/home/yuyao/methane_train/Upgrade_data_pipeline/csv/s2_6time_cdse_legacy512_exact}"
SOURCE_CSV="${SOURCE_CSV:-${CSV_ROOT}/s2_6time_cdse_legacy512_sources.csv}"
SOURCE_AUDIT="${SOURCE_AUDIT:-${CSV_ROOT}/s2_6time_cdse_legacy512_sources.audit.json}"
QA_512_CSV="${QA_512_CSV:-${CSV_ROOT}/s2_6time_cdse_legacy512_512_qa.csv}"
COMPLETE_512_CSV="${COMPLETE_512_CSV:-${CSV_ROOT}/s2_6time_cdse_legacy512_512_complete.csv}"
SPLIT_ROOT="${SPLIT_ROOT:-${CSV_ROOT}/temporal_split}"

OUT_512_ROOT="${OUT_512_ROOT:-/mnt/engg-niulab/yuyao/preprocessed_512/S2_6time_cdse_legacy512_exact}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-/diniuvol/yuyao/s2_6time_cdse_legacy512_exact_cache}"
OUT_32_ROOT="${OUT_32_ROOT:-${LOCAL_STAGE_ROOT}/patches_32}"
RESIZE_INPUT_32_ROOT="${RESIZE_INPUT_32_ROOT:-/mnt/engg-niulab/yuyao/final_crop/s2_6time_cdse_legacy512_exact_32}"
OUT_224_ROOT="${OUT_224_ROOT:-/mnt/engg-niulab/yuyao/final_crop/s2_6time_cdse_legacy512_exact_32_to_224}"

IMAGE_MODE="${IMAGE_MODE:-reuse-validated}"
BUILD_WORKERS="${BUILD_WORKERS:-16}"
CROP_WORKERS="${CROP_WORKERS:-16}"
RESIZE_WORKERS="${RESIZE_WORKERS:-8}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-${LOCAL_STAGE_ROOT}/plume_staging}"
RESIZE_CACHE_DIR="${RESIZE_CACHE_DIR:-${LOCAL_STAGE_ROOT}/resize_staging}"
CLEAN_LOCAL_STAGE="${CLEAN_LOCAL_STAGE:-0}"
PHASE="${1:-all}"

run_manifest() {
  "${PYTHON}" "${PIPELINE}" build-manifest \
    --out-512-root "${OUT_512_ROOT}" \
    --source-csv "${SOURCE_CSV}" \
    --source-audit-json "${SOURCE_AUDIT}" \
    --stat-workers 32
}

run_build_512() {
  "${PYTHON}" "${PIPELINE}" build-512 \
    --source-csv "${SOURCE_CSV}" \
    --qa-csv "${QA_512_CSV}" \
    --complete-csv "${COMPLETE_512_CSV}" \
    --image-mode "${IMAGE_MODE}" \
    --workers "${BUILD_WORKERS}" \
    --progress-every 50
}

run_split() {
  "${PYTHON}" "${PIPELINE}" split \
    --complete-csv "${COMPLETE_512_CSV}" \
    --split-root "${SPLIT_ROOT}" \
    --target-ratio 0.85 \
    --min-ratio 0.80 \
    --max-ratio 0.90
}

read_split_paths() {
  local split_audit="${SPLIT_ROOT}/split_audit.json"
  if [[ ! -s "${split_audit}" ]]; then
    echo "Missing split audit: ${split_audit}" >&2
    return 1
  fi
  mapfile -t SPLIT_PATHS < <(
    "${PYTHON}" -c \
      'import json,sys; d=json.load(open(sys.argv[1])); print(d["train_csv"]); print(d["test_csv"])' \
      "${split_audit}"
  )
  TRAIN_512_CSV="${SPLIT_PATHS[0]}"
  TEST_512_CSV="${SPLIT_PATHS[1]}"
}

run_crop() {
  read_split_paths
  "${PYTHON}" "${PIPELINE}" crop-32 \
    --train-csv "${TRAIN_512_CSV}" \
    --test-csv "${TEST_512_CSV}" \
    --out-32-root "${OUT_32_ROOT}" \
    --workers "${CROP_WORKERS}" \
    --local-cache-dir "${LOCAL_CACHE_DIR}" \
    --cache-copy-buffer-mb 8 \
    --no-cache-stage-outputs \
    --progress-every 250 \
    --resume
}

run_resize() {
  "${PYTHON}" "${PIPELINE}" resize-224 \
    --out-32-root "${RESIZE_INPUT_32_ROOT}" \
    --out-224-root "${OUT_224_ROOT}" \
    --workers "${RESIZE_WORKERS}" \
    --local-cache-dir "${RESIZE_CACHE_DIR}" \
    --cache-copy-buffer-mb 8 \
    --batch-files 512 \
    --progress-every 1000
}

run_audit() {
  "${PYTHON}" "${PIPELINE}" audit \
    --train-csv "${OUT_224_ROOT}/train_patches_224.csv" \
    --test-csv "${OUT_224_ROOT}/test_patches_224.csv" \
    --audit-json "${OUT_224_ROOT}/dataset_audit.json" \
    --path-audit-rows 1000 \
    --path-stat-workers 32
}

clean_local_stage() {
  if [[ "${CLEAN_LOCAL_STAGE}" != "1" ]]; then
    echo "Keeping local stage: ${LOCAL_STAGE_ROOT}"
    return 0
  fi
  case "${LOCAL_STAGE_ROOT}" in
    /diniuvol/yuyao/*)
      rm -rf -- "${LOCAL_STAGE_ROOT}"
      echo "Removed local stage: ${LOCAL_STAGE_ROOT}"
      ;;
    *)
      echo "Refusing to remove unsafe local stage: ${LOCAL_STAGE_ROOT}" >&2
      return 1
      ;;
  esac
}

case "${PHASE}" in
  manifest)
    run_manifest
    ;;
  build-512)
    run_build_512
    ;;
  split)
    run_split
    ;;
  crop-32)
    run_crop
    ;;
  resize-224)
    run_resize
    ;;
  audit)
    run_audit
    ;;
  clean-cache)
    clean_local_stage
    ;;
  all)
    run_manifest
    run_build_512
    run_split
    run_crop
    run_resize
    run_audit
    clean_local_stage
    ;;
  *)
    echo "Usage: $0 [manifest|build-512|split|crop-32|resize-224|audit|clean-cache|all]" >&2
    exit 2
    ;;
esac
