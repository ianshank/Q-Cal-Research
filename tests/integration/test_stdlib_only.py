"""Hook modules must import without site-packages, so guards work before any venv exists."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration

HOOK_MODULES = [
    "qcal.hooks.cli",
    "qcal.hooks.guards",
    "qcal.integrity.claims",
    "qcal.registry.tables",
    "qcal.registry.index",
    "qcal.policy",
    "qcal.config",
]


@pytest.mark.parametrize("module", HOOK_MODULES)
def test_module_imports_with_stdlib_only(module: str) -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); import {module}; "
        "assert 'yaml' not in sys.modules, 'yaml was imported'"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
