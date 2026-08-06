#!/usr/bin/env bash
#
# Run a sequence of "<side>:<seed>" replicates one at a time, each gated on
# GPU0 having room for a batch-32 finetune. Two chains can run side by side:
# when the card is busy they simply wait instead of OOMing each other.
#
# Usage: run_s2_multiseed_chain.sh <side:seed> [<side:seed> ...]

set -uo pipefail
root="/home/yuyao/panopticon/Upgraded_dataset"
for job in "$@"; do
  side="${job%%:*}"
  seed="${job##*:}"
  "${root}/run_s2_multiseed_when_free.sh" "${side}" "${seed}" || \
    echo "[chain $(date -u +%FT%TZ)] ${job} exited non-zero; continuing"
  sleep 60
done
