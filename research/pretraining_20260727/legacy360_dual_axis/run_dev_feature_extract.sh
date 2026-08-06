#!/usr/bin/env bash
set -euo pipefail

# Canonical development-only feature extraction. This launcher intentionally
# has no test argument; sealed test extraction is a separate one-use action.

repo=${REPO_ROOT:-/home/yuyao/panopticon}
python_bin=${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}
code=${repo}/research/pretraining_20260727
formal_root=${FORMAL_ROOT:-/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5}
manifest=${DEV_MANIFEST:-${code}/legacy360_dual_axis/sanitized_min1024/legacy360_dev_sanitized.csv}
expected_manifest_sha=${DEV_MANIFEST_SHA:-4f6fa26292461feb3def0213d97782c4968c4ad096394b5d94448ab23e7ed924}
output=${DEV_CACHE:-${formal_root}/features/dev_universal_s2hybrid.pt}
log=${DEV_LOG:-${formal_root}/logs/dev_feature_extract.log}
raw_cache=${RAW_CACHE:-/diniuvol/yuyao/methanefuse_research_20260727/legacy360_dual_axis/raw_cache}
universal_weights=${UNIVERSAL_WEIGHTS:-/transferdiniu2/yuyao/checkpoints/universal_360m/query dataset 360m/ckpt_best_test.pth}
s2_weights=${S2_WEIGHTS:-/transferdiniu2/yuyao/checkpoints/360m_single4_retrain_20260422_185153/s2/ckpt_best_test.pth}
device=${DEV_DEVICE:-cuda:1}

case "${manifest,,}" in
  *test*|*sealed*)
    echo "Refusing development manifest that looks like test/sealed data: ${manifest}" >&2
    exit 2
    ;;
esac

if [[ ! -f "$manifest" ]]; then
  echo "Missing canonical development manifest: $manifest" >&2
  exit 2
fi
observed_manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
if [[ "$observed_manifest_sha" != "$expected_manifest_sha" ]]; then
  echo "Canonical development manifest SHA mismatch: ${observed_manifest_sha}" >&2
  exit 2
fi
if [[ -e "$output" || -e "${output}.audit.json" ]]; then
  echo "Refusing existing development cache or audit: $output" >&2
  exit 3
fi

mkdir -p "$(dirname "$output")" "$(dirname "$log")"
echo "[dev-extract] manifest=${manifest}" | tee "$log"
echo "[dev-extract] output=${output} device=${device}" | tee -a "$log"
"$python_bin" "$code/query360_two_axis_full_legacy.py" extract \
  --manifest "$manifest" \
  --split dev \
  --output-cache "$output" \
  --weights "$universal_weights" \
  --sensor-weights "s2=${s2_weights}" \
  --raw-cache-dir "$raw_cache" \
  --raw-cache-readonly-fallback \
  --device "$device" \
  --amp-dtype float16 \
  --row-batch-size 64 \
  --encoder-microbatch 256 \
  --num-workers 2 \
  --prefetch-factor 1 \
  --log-interval 10 \
  2>&1 | tee -a "$log"
echo "[dev-extract] complete" | tee -a "$log"
