#!/usr/bin/env bash
# Nightly self-checks: deterministic, no agent involved. Cron or a scheduled CI job runs it.
#
#   scripts/nightly.sh                       # integrity checks only
#   NIGHTLY_GLOB='C-*fp32*' scripts/nightly.sh  # also run pending pre-registered cells
#
# Every check runs even if an earlier one fails; the exit status is non-zero if any failed.
# A JSON-lines report is written to $QCAL_NIGHTLY_DIR/<UTC date>.jsonl (default runs/nightly,
# which .gitignore keeps out of git). Environment:
#   QCAL            command that runs the CLI (default: python -m qcal)
#   QCAL_NIGHTLY_DIR  report directory
#   NIGHTLY_GLOB    if set, `qcal registry run-batch "$NIGHTLY_GLOB" --keep-going` runs first
#   NIGHTLY_MAX_RUNS  optional --max-runs for that batch
set -uo pipefail

root="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
cd "$root" || exit 2
read -r -a qcal <<<"${QCAL:-python -m qcal}"
report_dir="${QCAL_NIGHTLY_DIR:-runs/nightly}"
mkdir -p "$report_dir"
report="$report_dir/$(date -u +%Y-%m-%d).jsonl"
failures=0

json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }

check() {
  local name="$1"
  shift
  local started output status
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  output="$("$@" 2>&1)"
  status=$?
  if [ "$status" -ne 0 ]; then
    failures=$((failures + 1))
    echo "nightly: $name FAILED (exit $status)" >&2
  else
    echo "nightly: $name ok" >&2
  fi
  printf '{"check": "%s", "started": "%s", "exit": %d, "output": %s}\n' \
    "$name" "$started" "$status" "$(printf '%s' "$output" | json_escape)" >>"$report"
}

if [ -n "${NIGHTLY_GLOB:-}" ]; then
  batch=(registry run-batch "$NIGHTLY_GLOB" --keep-going)
  if [ -n "${NIGHTLY_MAX_RUNS:-}" ]; then
    batch+=(--max-runs "$NIGHTLY_MAX_RUNS")
  fi
  check run-batch "${qcal[@]}" "${batch[@]}"
  check index "${qcal[@]}" registry index
fi

check agent-layer "${qcal[@]}" agent-layer
check index-current "${qcal[@]}" registry index --check
check tables-current "${qcal[@]}" registry tables --check
check audit "${qcal[@]}" registry audit --json
check claims "${qcal[@]}" claims
check leakage "${qcal[@]}" leakage --if-present
check licenses "${qcal[@]}" licenses

echo "nightly: $failures failing check(s); report in $report" >&2
[ "$failures" -eq 0 ]
