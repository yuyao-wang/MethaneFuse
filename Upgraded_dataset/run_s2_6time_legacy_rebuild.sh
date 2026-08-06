#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "This runner now uses the verified CDSE raw + engg-leung legacy-512 pipeline." >&2
exec "${SCRIPT_DIR}/run_s2_6time_cdse_legacy512_rebuild.sh" "$@"
