#!/usr/bin/env bash
# SessionStart hook: bootstrap the Python environment in Claude Code cloud sessions.
#
# Synchronous by design, so tests and linters work from the first prompt.
# Installs only when CLAUDE_CODE_REMOTE=true; locally it just reports facts.
# Idempotent: re-running reuses .venv and pip's cache. Overridable via env:
#   QCAL_PYTHON   interpreter used to create the venv (default: python3)
#   QCAL_VENV     venv directory relative to the project (default: .venv)
#   QCAL_EXTRAS   pip extras to install (default: dev)
set -euo pipefail

script_dir="${BASH_SOURCE[0]%/*}"
root="${CLAUDE_PROJECT_DIR:-$(cd "$script_dir/../.." && pwd)}"
venv="$root/${QCAL_VENV:-.venv}"
python_bin="${QCAL_PYTHON:-python3}"
extras="${QCAL_EXTRAS:-dev}"

if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then
  if [ ! -x "$venv/bin/python" ]; then
    "$python_bin" -m venv "$venv"
  fi
  "$venv/bin/python" -m pip install --quiet --disable-pip-version-check --upgrade pip
  "$venv/bin/python" -m pip install --quiet --disable-pip-version-check -e "$root[$extras]"
  if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    {
      echo "export VIRTUAL_ENV=\"$venv\""
      echo "export PATH=\"$venv/bin:\$PATH\""
    } >> "$CLAUDE_ENV_FILE"
  fi
fi

# Facts for Claude's context (stdout of SessionStart is added to it). Never fails the hook.
if [ -x "$venv/bin/python" ]; then
  "$venv/bin/python" - <<'PY' 2>/dev/null || true
import importlib.util, platform
facts = [f"python {platform.python_version()}"]
if importlib.util.find_spec("torch"):
    import torch
    archs = " ".join(torch.cuda.get_arch_list()) if torch.cuda.is_available() else "no CUDA"
    facts.append(f"torch {torch.__version__} (cuda {torch.version.cuda}; {archs})")
print("qcal environment: " + "; ".join(facts))
PY
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
