#!/usr/bin/env bash
# Fail-closed launcher for qcal Claude Code hooks.
#
#   run_hook.sh [--fail-open] <hook-name> [args...]
#
# Claude Code treats only exit code 2 as "block"; a crash (exit 1) would let the
# tool call through. This wrapper converts every unexpected failure into exit 2,
# including a missing interpreter. --fail-open (used for the Stop hook) converts
# failures into exit 0 instead, because CI runs the same check as the real gate.
#
# Interpreter: $QCAL_PYTHON, else the project .venv, else python3 (stdlib only is needed).
set -u

fail_code=2
label="BLOCKED"
if [ "${1:-}" = "--fail-open" ]; then
  fail_code=0
  label="WARNING"
  shift
fi
hook="${1:-}"
if [ -z "$hook" ]; then
  echo "$label: run_hook.sh needs a hook name" >&2
  exit "$fail_code"
fi

script_dir="${BASH_SOURCE[0]%/*}"
root="${CLAUDE_PROJECT_DIR:-$(cd "$script_dir/../.." && pwd)}"
python_bin="${QCAL_PYTHON:-}"
if [ -z "$python_bin" ]; then
  for candidate in "$root/.venv/bin/python" python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi
if [ -z "$python_bin" ]; then
  echo "$label: qcal hook '$hook' cannot run: no Python interpreter found" >&2
  exit "$fail_code"
fi

PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -s -m qcal.hooks "$@"
status=$?
case "$status" in
  0|2) exit "$status" ;;
esac
echo "$label: qcal hook '$hook' exited with status $status; treating it as a failure" >&2
exit "$fail_code"
