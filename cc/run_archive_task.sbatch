#!/usr/bin/env bash
#SBATCH --account=def-juliana2_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --tmp=3000G
#SBATCH --time=24:00:00
#SBATCH --job-name=panopticon-query
#SBATCH --output=/scratch/yuyao16/panopticon_logs/%x-%j.out

set -euo pipefail

: "${DATASET:?Set DATASET, e.g. legacy_param_960m}"
: "${TASK:?Set TASK to query_all, query_universal, query_seg, single, or seg}"

REPO=${REPO:-$HOME/panopticon_methane_detector}
ARCHIVE=${ARCHIVE:-/scratch/yuyao16/panopticon_archives/${DATASET}.tar}
ARCHIVE_PART_GLOB=${ARCHIVE_PART_GLOB:-/scratch/yuyao16/panopticon_archives/${DATASET}.part*.tar}
OLD_ROOT=${OLD_ROOT:-/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/${DATASET}}
RUN_LABEL=${RUN_LABEL:-query dataset ${DATASET#legacy_param_}}
OUTPUT_BASE=${OUTPUT_BASE:-/scratch/yuyao16/panopticon_outputs}
WEIGHTS=${WEIGHTS:-$REPO/weights/panopticon_vitb14_teacher.pth}

module load python/3.11 cuda/12.2

if [ -n "${VENV:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

cd "$REPO"

[ -f "$WEIGHTS" ] || { echo "Missing weights: $WEIGHTS" >&2; exit 1; }

WORK_ROOT="$SLURM_TMPDIR/panopticon_data"
DATA_ROOT="$WORK_ROOT/$DATASET"
mkdir -p "$DATA_ROOT" "$SLURM_TMPDIR/cache" "$SLURM_TMPDIR/tmp"

shopt -s nullglob
archive_parts=($ARCHIVE_PART_GLOB)
shopt -u nullglob
if [ -f "$ARCHIVE" ]; then
  echo "[extract] $ARCHIVE -> $DATA_ROOT"
  tar -C "$DATA_ROOT" -xf "$ARCHIVE"
elif [ "${#archive_parts[@]}" -gt 0 ]; then
  printf "[extract] %d archive parts -> %s\n" "${#archive_parts[@]}" "$DATA_ROOT"
  for part in "${archive_parts[@]}"; do
    echo "[extract-part] $part"
    tar -C "$DATA_ROOT" -xf "$part"
  done
else
  echo "Missing archive: $ARCHIVE or parts matching $ARCHIVE_PART_GLOB" >&2
  exit 1
fi
python cc/rewrite_manifest_paths.py --data-root "$DATA_ROOT" --old-root "$OLD_ROOT"

export DATA_ROOT
export TRAIN_CSV="$DATA_ROOT/manifest_time_train.csv"
export TEST_CSV="$DATA_ROOT/manifest_time_test.csv"
export TMP_BASE="$SLURM_TMPDIR/tmp"
export CACHE_DIR="$SLURM_TMPDIR/cache"
export WEIGHTS
export RUN_LABEL
export PYTHON_BIN=python
export OUT_ROOT="$OUTPUT_BASE/$DATASET/$TASK/$SLURM_JOB_ID"
mkdir -p "$OUT_ROOT"

case "$TASK" in
  query_all)
    export QUERY_MODE=all
    export UNIVERSAL_CHECKPOINT_DIR="$OUT_ROOT/universal"
    export SEG_CHECKPOINT_DIR="$OUT_ROOT/universal_seg"
    bash run_query.sh
    ;;
  query_universal)
    export QUERY_MODE=universal
    export UNIVERSAL_CHECKPOINT_DIR="$OUT_ROOT/universal"
    bash run_query.sh
    ;;
  query_seg)
    : "${UNIVERSAL_CKPT:?Set UNIVERSAL_CKPT for TASK=query_seg}"
    export QUERY_MODE=seg
    export SEG_CHECKPOINT_DIR="$OUT_ROOT/universal_seg"
    bash run_query.sh
    ;;
  single)
    bash run_single.sh
    ;;
  seg)
    bash run_seg.sh
    ;;
  *)
    echo "Unknown TASK: $TASK" >&2
    exit 2
    ;;
esac
