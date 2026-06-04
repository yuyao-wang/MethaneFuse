#!/usr/bin/env bash
set -euo pipefail

# Adapter fine-tuning entry point kept for backward-compatible script naming.
exec "$(dirname "$0")/run_fuse.sh" "$@"
