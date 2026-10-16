"""scripts/nightly.sh: every check runs, a JSON-lines report is written, failures set the exit."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from qcal.cli import main
from tests.conftest import REPO_ROOT, experiments_yaml, write

SCRIPT = REPO_ROOT / "scripts" / "nightly.sh"
CHECKS = [
    "agent-layer",
    "index-current",
    "tables-current",
    "audit",
    "claims",
    "leakage",
    "licenses",
]


def run_nightly(project: Path, reports: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    qcal = f"{shlex.quote(sys.executable)} -m qcal --root {shlex.quote(str(project))}"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "QCAL": qcal,
        "QCAL_NIGHTLY_DIR": str(reports),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        **extra,
    }
    return subprocess.run(
        [str(SCRIPT)], capture_output=True, text=True, env=env, check=False, timeout=300
    )


def entries(reports: Path) -> list[dict[str, object]]:
    (report,) = reports.glob("*.jsonl")
    return [json.loads(line) for line in report.read_text().splitlines()]


def test_every_check_runs_and_failures_are_reported(repo: Path, tmp_path: Path) -> None:
    reports = tmp_path / "nightly"

    result = run_nightly(repo, reports)

    assert result.returncode == 1  # no EXPERIMENTS.yaml yet: the audit cannot run
    rows = entries(reports)
    assert [r["check"] for r in rows] == CHECKS
    assert {r["check"] for r in rows if r["exit"] != 0} == {"audit"}
    assert "nightly: 1 failing check(s)" in result.stderr


def test_a_clean_project_passes(repo: Path, tmp_path: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    assert main(["--root", str(repo), "registry", "index"]) == 0
    reports = tmp_path / "nightly"

    result = run_nightly(repo, reports)

    assert result.returncode == 0, result.stderr
    assert all(r["exit"] == 0 for r in entries(reports))


@pytest.mark.parametrize("max_runs", ["", "1"])
def test_nightly_glob_runs_pending_cells_first(repo: Path, tmp_path: Path, max_runs: str) -> None:
    program = tmp_path / "experiment.py"
    program.write_text(
        "import json, sys\n"
        "ENVELOPE = {'format': 'qcal.executor_result', 'version': 1, 'status': 'ok'}\n"
        "json.dump({**ENVELOPE, 'metrics': {'AP': 40.0}}, open(sys.argv[1], 'w'))\n",
        encoding="utf-8",
    )
    command = json.dumps([sys.executable, str(program), "{result_path}"])
    write(repo, "qcal.toml", f"[executor]\ncommand = {command}\n")
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    reports = tmp_path / "nightly"
    extra = {"NIGHTLY_GLOB": "C-*"} | ({"NIGHTLY_MAX_RUNS": max_runs} if max_runs else {})

    result = run_nightly(repo, reports, **extra)

    rows = entries(reports)
    assert [r["check"] for r in rows][:2] == ["run-batch", "index"]
    assert rows[0]["exit"] == 0, rows[0]
    assert result.returncode == 0, result.stderr
    assert len(list((repo / "runs" / "registry").glob("*.json"))) == 1
    assert next(r for r in rows if r["check"] == "index-current")["exit"] == 0
