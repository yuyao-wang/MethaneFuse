#!/usr/bin/env bash
#
# Hand GPU0 over to a re-read priority config without disturbing the stage that
# is already training.
#
# run_s2_priority_sequence.py parses its config once at startup, so editing the
# JSON does not change what an already-running supervisor will launch next. To
# swap a queued stage we therefore stop only the supervisor process, let its
# current convergence-queue child finish on its own, and start a fresh
# supervisor that picks up the edited config.
#
# Re-running the whole config is safe: run_s2_convergence_queue.py reports a
# split as already_complete when it has reached max_epochs or recorded
# early_stopped, so finished stages are skipped rather than retrained.
#
# Usage:
#   run_s2_gpu0_queue_swap.sh <queue_pid_to_wait_for>

set -euo pipefail

wait_pid="${1:?usage: $0 <queue_pid_to_wait_for>}"

root="/home/yuyao/panopticon/Upgraded_dataset"
python_bin="/home/yuyao/miniconda3/envs/panopticon/bin/python"
config="${root}/s2_priority_gpu0_legacy.json"
status="${root}/s2_legacy360_matched_gee_splits/priority_gpu0_status.json"
log="${root}/s2_legacy360_matched_gee_splits/priority_gpu0_swap.log"

# The trainer is always invoked with --device cuda:0, so the physical GPU is
# selected purely by CUDA_VISIBLE_DEVICES inherited from this supervisor. Pin it
# explicitly: this queue owns GPU0 while the separate GPU1 supervisor keeps
# CUDA_VISIBLE_DEVICES=1.
export CUDA_VISIBLE_DEVICES=0

exec >>"${log}" 2>&1
echo "[swap $(date -u +%FT%TZ)] waiting for queue pid ${wait_pid} to exit"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 60
done

echo "[swap $(date -u +%FT%TZ)] pid ${wait_pid} exited; starting supervisor on ${config}"
exec "${python_bin}" "${root}/run_s2_priority_sequence.py" \
  --config "${config}" \
  --status "${status}"
