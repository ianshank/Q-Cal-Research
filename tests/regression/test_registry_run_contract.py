"""Regression tests for the registered-run contract (docs/changes/registry-run-contract.md).

One test per gap the proposal names; each fails on the code before its fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from qcal.config import load_config
from qcal.registry.executor import RunSpec, SubprocessExecutor
from qcal.registry.experiments import Cell, load_experiments
from qcal.registry.runner import Runner
from qcal.registry.store import RegistryStore
from tests.conftest import REPO_ROOT, executor_command, experiment_script, experiments_yaml, write

pytestmark = pytest.mark.rule("C7")


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    command = json.dumps(executor_command(experiment_script(tmp_path)))
    (root / "qcal.toml").write_text(f"[executor]\ncommand = {command}\n", encoding="utf-8")
    write(root, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    return root


def _runner(root: Path) -> Runner:
    config = load_config(root, environ={})
    return Runner(
        config,
        load_experiments(config),
        RegistryStore(config.path("registry_dir")),
        SubprocessExecutor.from_config(config),
        collectors=["python"],
    )


def test_the_repository_command_runs_the_launchers_interpreter(tmp_path: Path) -> None:
    """Gap: the command ran whatever `python3` was first on PATH."""
    executor = SubprocessExecutor.from_config(load_config(REPO_ROOT, environ={}))
    spec = RunSpec(
        run_id="R1",
        cell=Cell(id="C-a", factors={}),
        seed=0,
        seed_role="calibrator_fit_draw",
        result_path=tmp_path / "R1.json",
        log_path=tmp_path / "R1.log",
        root=tmp_path,
        python="/opt/launcher/bin/python",
    )
    assert executor.argv(spec)[:2] == ["/opt/launcher/bin/python", "-I"]


def test_a_record_names_its_interpreter_variables_and_launcher_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gaps: no interpreter recorded, no CUDA variables, launcher and program keys mixed."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa")
    record = _runner(_project(tmp_path)).run("C-a", 0)
    assert record.status == "ok", record.error
    assert record.provenance["launcher"]["python"] == sys.executable
    assert record.provenance["environment_variables"]["CUDA_VISIBLE_DEVICES"] == "GPU-aaaa"
    assert record.provenance["environment_variables"]["CUDA_DEVICE_ORDER"] is None
    assert "launcher.python" in record.environment
    assert record.environment["trt_version"] == "test"  # the program's own key, unprefixed
