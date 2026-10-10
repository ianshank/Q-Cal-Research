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
facts_python="$python_bin"
if [ -x "$venv/bin/python" ]; then
  facts_python="$venv/bin/python"
fi
PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" "$facts_python" -P - <<'PY' 2>/dev/null || true
import importlib.util, os, platform
facts = [f"python {platform.python_version()}"]
try:
    from qcal.config import load_config
    from qcal.hooks.decision import resolve_mode

    config = load_config(environ=os.environ)
    facts.append(f"guard mode {resolve_mode(config, os.environ).value}")
    facts.append(f"signing.mode {config.str_value('signing.mode')}")
    facts.append(f"review.mode {config.str_value('review.mode')}")
except Exception as exc:  # report, never fail the session
    facts.append(f"qcal configuration unavailable ({type(exc).__name__}: {exc})")
if importlib.util.find_spec("torch"):
    import torch
    archs = " ".join(torch.cuda.get_arch_list()) if torch.cuda.is_available() else "no CUDA"
    facts.append(f"torch {torch.__version__} (cuda {torch.version.cuda}; {archs})")
print("qcal environment: " + "; ".join(facts))
PY
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
