"""``make smoke``: the Phase 1 loop on the synthetic fixture, within its time budget."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from qcal_lab.smoke import smoke_settings
from tests.conftest import REPO_ROOT, require_tool

pytestmark = pytest.mark.e2e


def test_python_m_qcal_lab_smoke_passes_within_budget() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "qcal_lab", "--root", str(REPO_ROOT), "smoke", "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    report = json.loads(out.stdout)
    assert out.returncode == 0, report
    budget = float(smoke_settings()["max_seconds"])
    assert report["seconds"] < budget
    names = [c["name"] for c in report["checks"]]
    assert (
        "an uncached re-run reproduces predictions, calibration and metrics byte for byte" in names
    )


def test_make_smoke() -> None:
    make = require_tool("make")
    out = subprocess.run(
        [make, "--no-print-directory", "smoke", f"PYTHON={sys.executable}"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "\nsmoke: PASS" in out.stdout  # make echoes the command first
