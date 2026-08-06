#!/usr/bin/env bash
set -euo pipefail

# Resume-safe train_core feature extraction. Each GPU processes its own
# event-safe manifest chunks sequentially; completed atomic .pt files are
# skipped on restart. This launcher has no held-out/test input.

if [[ $# -lt 1 || $# -gt 2 ]] || [[ ! "$1" =~ ^[01]$ ]]; then
  echo "Usage: $0 <gpu-id: 0|1> [two-digit-chunk-index]" >&2
  exit 2
fi

gpu_id=$1
only_chunk=${2:-}
if [[ -n "$only_chunk" && ! "$only_chunk" =~ ^[0-9][0-9]$ ]]; then
  echo "Optional chunk index must be exactly two digits; got: $only_chunk" >&2
  exit 2
fi
repo=${REPO_ROOT:-/home/yuyao/panopticon}
python_bin=${PYTHON_BIN:-/home/yuyao/miniconda3/envs/panopticon/bin/python}
code=${repo}/research/pretraining_20260727
chunk_root=${CHUNK_ROOT:-${code}/legacy360_dual_axis/manifests/chunks}
formal_root=${FORMAL_ROOT:-/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5}
part_root=${formal_root}/features/parts
log_root=${formal_root}/logs/chunks
raw_cache=${RAW_CACHE:-/diniuvol/yuyao/methanefuse_research_20260727/legacy360_dual_axis/raw_cache}
universal_weights=${UNIVERSAL_WEIGHTS:-/transferdiniu2/yuyao/checkpoints/universal_360m/query dataset 360m/ckpt_best_test.pth}
s2_weights=${S2_WEIGHTS:-/transferdiniu2/yuyao/checkpoints/360m_single4_retrain_20260422_185153/s2/ckpt_best_test.pth}
num_workers=${NUM_WORKERS:-2}
prefetch_factor=${PREFETCH_FACTOR:-1}
row_batch_size=${ROW_BATCH_SIZE:-64}
encoder_microbatch=${ENCODER_MICROBATCH:-256}

if [[ ! "$num_workers" =~ ^[1-8]$ ]]; then
  echo "NUM_WORKERS must be between 1 and 8; got: $num_workers" >&2
  exit 2
fi
if [[ ! "$prefetch_factor" =~ ^[1-4]$ ]]; then
  echo "PREFETCH_FACTOR must be between 1 and 4; got: $prefetch_factor" >&2
  exit 2
fi

mkdir -p "$part_root" "$log_root"
shopt -s nullglob
if [[ -n "$only_chunk" ]]; then
  manifests=("${chunk_root}"/legacy360_train_core_gpu"${gpu_id}"_chunk"${only_chunk}".csv)
else
  manifests=("${chunk_root}"/legacy360_train_core_gpu"${gpu_id}"_chunk*.csv)
fi
if [[ ${#manifests[@]} -eq 0 ]]; then
  echo "No manifests found for GPU ${gpu_id} under ${chunk_root}" >&2
  exit 2
fi

for manifest in "${manifests[@]}"; do
  stem=$(basename "$manifest" .csv)
  output=${part_root}/${stem}.pt
  log=${log_root}/${stem}.log
  if [[ -s "$output" && -s "${output}.audit.json" ]]; then
    echo "[chunk-launch] gpu=${gpu_id} skip completed ${stem}" | tee -a "$log"
    continue
  fi
  if [[ -e "$output" || -e "${output}.audit.json" ]]; then
    echo "Refusing partial/existing output for ${stem}: ${output}" >&2
    exit 3
  fi
  echo "[chunk-launch] gpu=${gpu_id} start ${stem}" | tee "$log"
  "$python_bin" "$code/query360_two_axis_full_legacy.py" extract \
    --manifest "$manifest" \
    --split train_core \
    --output-cache "$output" \
    --weights "$universal_weights" \
    --sensor-weights "s2=${s2_weights}" \
    --raw-cache-dir "$raw_cache" \
    --raw-cache-readonly-fallback \
    --device "cuda:${gpu_id}" \
    --amp-dtype float16 \
    --row-batch-size "$row_batch_size" \
    --encoder-microbatch "$encoder_microbatch" \
    --num-workers "$num_workers" \
    --prefetch-factor "$prefetch_factor" \
    --log-interval 10 \
    2>&1 | tee -a "$log"
  echo "[chunk-launch] gpu=${gpu_id} complete ${stem}" | tee -a "$log"
done

echo "[chunk-launch] gpu=${gpu_id} all chunks complete"
