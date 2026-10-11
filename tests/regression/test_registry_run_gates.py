"""Regressions for the cycle-2026-10 audit, PR-A1 (registered-run gates).

A1: configuration from the environment changed a registered run without a trace.
A2: ``--experiments`` recorded one file's factors while the program ran another's.
A3: the experiment program re-read configuration that could differ from the launcher's.
A4: a misspelled policy key fell back to its default silently.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qcal import cli
from qcal.config import load_config
from qcal.registry.gates import INPUTS_READ_KEY
from qcal_lab.experiment import PlanError, RunRequest, run_experiment
from tests.conftest import REPO_ROOT, experiments_yaml, write
from tests.lab_support import fixture_project

pytestmark = pytest.mark.rule("C2", "C7")


def _run(root: Path, *args: str, environ: dict[str, str]) -> int:
    return cli.main(["--root", str(root), *args], out=io.StringIO(), environ=environ)


def test_a1_environment_overrides_are_refused_and_logging_ones_recorded(repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    script = repo / "prog.py"
    script.write_text(
        "import json, sys\njson.dump({'metrics': {'AP': 1.0}}, open(sys.argv[1], 'w'))\n"
    )
    (repo / "qcal.toml").write_text(
        f'[executor]\ncommand = ["{sys.executable}", "{script}", "{{result_path}}"]\n'
        "[registry]\nenv_collectors = []\n"
    )
    refused = {"QCAL__PATHS__MANIFESTS_DIR": "/tmp/elsewhere"}
    assert _run(repo, "registry", "run", "C-a", "--seed", "0", environ=refused) == 2
    assert not (repo / "runs/registry").exists()
    neutral = {"QCAL__LOGGING__LEVEL": "WARNING"}
    out = io.StringIO()
    code = cli.main(
        ["--root", str(repo), "registry", "run", "C-a", "--seed", "0", "--json"],
        out=out,
        environ=neutral,
    )
    assert code == 0
    assert json.loads(out.getvalue())["provenance"]["config_environment"] == neutral


def test_a2_experiments_must_be_the_configured_file(repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a", "detector": "atss"}]))
    write(repo, "other.yaml", experiments_yaml([{"id": "C-a", "detector": "swapped"}]))
    args = ("registry", "run", "C-a", "--seed", "0", "--experiments", str(repo / "other.yaml"))
    assert _run(repo, *args, environ={}) == 2


def test_a3_the_program_refuses_environment_configuration(tmp_path: Path) -> None:
    root, cells = fixture_project(tmp_path)
    request = RunRequest(root, "R-env", cells[0], 0, root / "runs/results/R-env.json")
    config = load_config(root, environ={"QCAL__DATA__SPLITS": '["val", "test"]'})
    with pytest.raises(PlanError, match="QCAL__DATA__SPLITS overrides configuration"):
        run_experiment(request, qcal_config=config)


def test_a3_the_program_reports_what_it_read(tmp_path: Path) -> None:
    root, cells = fixture_project(tmp_path)
    request = RunRequest(root, "R-read", cells[0], 0, root / "runs/results/R-read.json")
    reported = run_experiment(request).environment[INPUTS_READ_KEY]
    assert set(reported) == {"EXPERIMENTS.yaml", "qcal.toml", "configs/lab.toml"}
    import hashlib

    for path, digest in reported.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path


def test_a4_a_misspelled_policy_key_fails_config_check(repo: Path) -> None:
    (repo / "qcal.toml").write_text('[signing]\nsigned_categorys = ["ian_only"]\n')
    assert _run(repo, "config", "--check", environ={}) == 1


def test_a4_the_repository_policy_passes_config_check() -> None:
    assert _run(REPO_ROOT, "config", "--check", environ={}) == 0


@pytest.mark.e2e
def test_a1_smoke_ignores_configuration_variables_in_the_callers_shell(tmp_path: Path) -> None:
    """``QCAL__REGISTRY__MAX_RUNS_PER_BATCH=1`` would refuse the six-cell batch if it leaked."""
    env = {
        **os.environ,
        "QCAL__REGISTRY__MAX_RUNS_PER_BATCH": "1",
        "QCAL__DATA__SPLITS": '["val"]',
    }
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "qcal_lab",
            "--root",
            str(REPO_ROOT),
            "smoke",
            "--json",
            "--workdir",
            str(tmp_path / "w"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env=env,
    )
    report = json.loads(out.stdout)
    assert out.returncode == 0, report
