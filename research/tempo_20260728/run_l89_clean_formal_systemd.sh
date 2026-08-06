#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yuyao/panopticon"
launcher="$repo_root/research/tempo_20260728/run_l89_clean_replicate.sh"
formal_root="/diniuvol/yuyao/methanefuse_research_20260728/l89_clean_replicate_run_v1"
unit_name="tempo-l89-clean-formal-v1"

command=(
  systemd-run
  --user
  --collect
  --unit="$unit_name"
  --description="Clean train-only L89 TEMPO replicate on physical GPU0"
  --property=Type=exec
  --property="WorkingDirectory=$repo_root"
  --property=MemoryMax=96G
  --property=CPUQuota=3000%
  --property=TasksMax=4096
  --property=OOMPolicy=stop
  --property=TimeoutStopSec=60
  --setenv="PYTHONPATH=$repo_root"
  /usr/bin/bash
  "$launcher"
  --run
)

if [[ "${1:-}" != "--run" ]]; then
  printf 'Dry run only. Formal output root must not exist: %s\n' "$formal_root"
  printf 'Exact persistent launch command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi
shift
if [[ "$#" -ne 0 ]]; then
  printf 'Unexpected arguments: %s\n' "$*" >&2
  exit 2
fi
if [[ -e "$formal_root" ]]; then
  printf 'Refusing existing formal root: %s\n' "$formal_root" >&2
  exit 3
fi
if [[ ! -f "$launcher" ]]; then
  printf 'Formal launcher is missing: %s\n' "$launcher" >&2
  exit 4
fi

"${command[@]}"
