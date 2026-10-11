"""The documented entry points start in a fresh, isolated interpreter (``python -I``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from qcal import __version__
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration


def run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, cwd=REPO_ROOT)


def test_python_m_qcal_reports_its_version() -> None:
    out = run(sys.executable, "-I", "-m", "qcal", "--version")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == f"qcal {__version__}"


@pytest.mark.parametrize("module", ["qcal.hooks", "qcal_lab"])
def test_python_m_modules_print_help(module: str) -> None:
    out = run(sys.executable, "-I", "-m", module, "--help")
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout


@pytest.mark.parametrize("script", ["qcal", "qcal-registry", "qcal-claims"])
def test_console_scripts_are_installed_and_start(script: str) -> None:
    path = Path(sys.executable).with_name(script)
    if not path.exists():
        pytest.skip(f"{script} is not installed next to {sys.executable}")
    out = run(str(path), "--help")
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout
