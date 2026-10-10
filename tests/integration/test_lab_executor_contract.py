"""``python -m qcal_lab run`` honours the registry's executor contract in a real subprocess."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from qcal.config import load_config
from qcal.registry.executor import RunSpec, SubprocessExecutor
from qcal.registry.experiments import load_experiments
from tests.conftest import REPO_ROOT
from tests.lab_support import fixture_project

pytestmark = pytest.mark.integration


def _spec(project: Path, cell_id: str, run_id: str) -> RunSpec:
    config = load_config(project, environ={})
    cell = load_experiments(config).cell(cell_id)
    return RunSpec(
        run_id=run_id,
        cell=cell,
        seed=0,
        seed_role="calibrator_fit_draw",
        result_path=project / "runs/results" / f"{run_id}.json",
        log_path=project / "runs/logs" / f"{run_id}.log",
        root=project,
    )


def _executor(project: Path) -> SubprocessExecutor:
    config = load_config(project, environ={})
    return SubprocessExecutor(
        config.str_list("executor.command"), timeout_s=120, env_prefix="QCAL_RUN_", root=project
    )


def test_a_cell_runs_through_the_subprocess_executor(tmp_path: Path) -> None:
    project, cells = fixture_project(tmp_path)
    result = _executor(project).execute(_spec(project, cells[3], "R-int"))
    assert result.ok, result.error
    assert set(result.metrics) == {"smoke_images", "smoke_detections", "smoke_mean_abs_gap"}
    assert [a.kind for a in result.artifacts][-2:] == ["calibration", "predictions_calibrated_test"]
    assert all(len(a.sha256) == 64 for a in result.artifacts)
    assert result.environment["eval_loop"]["module"] == "qcal_lab.fixture_eval:build"


def test_a_failing_cell_becomes_a_failed_result_with_its_log(tmp_path: Path) -> None:
    project, cells = fixture_project(tmp_path)
    text = (project / "EXPERIMENTS.yaml").read_text().replace("precision: fp32", "precision: int8")
    (project / "EXPERIMENTS.yaml").write_text(text)
    spec = _spec(project, cells[0], "R-bad")
    result = _executor(project).execute(spec)
    assert not result.ok
    assert result.error is not None
    assert "exit code 1" in result.error
    assert "precision='int8' is not supported" in spec.log_path.read_text()


def test_the_repository_wires_the_experiment_program() -> None:
    config = load_config(REPO_ROOT, environ={})
    command = config.str_list("executor.command")
    assert command[:5] == ["python3", "-I", "-m", "qcal_lab", "--root"]
    assert {"{run_id}", "{cell_id}", "{seed}", "{result_path}"} <= set(command)


def test_the_module_runs_as_a_script(tmp_path: Path) -> None:
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "qcal_lab", "--root", str(tmp_path), "status", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["ready"] is False
