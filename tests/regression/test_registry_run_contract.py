"""Regression tests for the registered-run contract (docs/changes/registry-run-contract.md).

One test per gap the proposal names; each fails on the code before its fix.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qcal.config import load_config, without_config_environment
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


FAILS_WITH_AN_ENVELOPE = """
import json, sys
json.dump({"format": "qcal.executor_result", "version": 1, "status": "failed",
           "failure_kind": "cuda_oom", "error": "CUDA out of memory", "metrics": {"AP": 1.0},
           "environment": {"torch": "2.9"}}, open(sys.argv[1], "w"))
sys.exit(1)
"""


def test_a_failed_run_records_why_and_what_the_program_saw(tmp_path: Path) -> None:
    """Gap: a failed record kept neither the program's environment nor a failure kind."""
    program = tmp_path / "fails.py"
    program.write_text(FAILS_WITH_AN_ENVELOPE, encoding="utf-8")
    root = _project(tmp_path)
    command = json.dumps([sys.executable, str(program), "{result_path}"])
    (root / "qcal.toml").write_text(f"[executor]\ncommand = {command}\n", encoding="utf-8")
    record = _runner(root).run("C-a", 0)
    assert (record.status, record.failure_kind) == ("failed", "cuda_oom")
    assert record.error is not None
    assert record.error.endswith(": CUDA out of memory")
    assert dict(record.metrics) == {}
    assert record.environment["torch"] == "2.9"


def test_a_record_says_what_the_run_cost(tmp_path: Path) -> None:
    """Gap: a record had duration_s only, no CPU time or memory."""
    record = _runner(_project(tmp_path)).run("C-a", 0)
    assert {"wall_s", "cpu_user_s", "cpu_sys_s", "peak_rss_kib"} <= set(record.resources)
    assert record.resources["peak_rss_kib"] > 0


def test_tables_refuse_to_average_runs_under_different_configuration_files(
    tmp_path: Path,
) -> None:
    """Gap: tables averaged seeds without checking they ran under the same configuration."""
    from qcal.registry.index import write_index
    from qcal.registry.tables import TableDataError, build_tables

    root = _project(tmp_path)
    write(root, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0, 1]))
    write(root, "configs/lab.toml", "x = 1\n")
    _runner(root).run("C-a", 0)
    write(root, "configs/lab.toml", "x = 2\n")  # a science setting changed between seeds
    _runner(root).run("C-a", 1)
    config = load_config(root, environ={})
    write_index(config, RegistryStore(config.path("registry_dir")))
    write(
        root,
        "configs/tables/t.toml",
        '[[table]]\nname = "t"\nrows = ["cell_id"]\n[[table.columns]]\nmetric = "AP"\n',
    )
    with pytest.raises(TableDataError, match="different configuration files"):
        build_tables(config)


SLEEPS_AND_REPORTS_ITS_PID = """
import os, sys, time
open(sys.argv[1], "w").write(str(os.getpid()))
time.sleep(60)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/stat")
    return not (status.is_file() and status.read_text().split()[2] == "Z")


@pytest.mark.integration
@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc (Linux)")
@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGHUP], ids=["sigterm", "sighup"])
def test_a_killed_launcher_takes_its_program_down_and_records_it(
    tmp_path: Path, number: int
) -> None:
    """Gap (wave-2 review B6): the program runs in its own session, so `kill` or a closed
    terminal ended the launcher and left the program running, unrecorded, holding the GPU."""
    program = tmp_path / "sleeper.py"
    program.write_text(SLEEPS_AND_REPORTS_ITS_PID, encoding="utf-8")
    pid_file = tmp_path / "program.pid"
    root = tmp_path / "project"
    root.mkdir()
    command = json.dumps([sys.executable, str(program), str(pid_file)])
    (root / "qcal.toml").write_text(f"[executor]\ncommand = {command}\n", encoding="utf-8")
    write(root, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "qcal",
            "--root",
            str(root),
            "registry",
            "run",
            "C-a",
            "--seed",
            "0",
        ],
        env=without_config_environment(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while not (pid_file.is_file() and pid_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    program_pid = int(pid_file.read_text())
    launcher.send_signal(number)
    assert launcher.wait(timeout=60) == 130
    deadline = time.monotonic() + 10
    while _alive(program_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(program_pid)
    config = load_config(root, environ={})
    (record,) = RegistryStore(config.path("registry_dir")).load_all()
    assert (record.status, record.failure_kind) == ("failed", "interrupted")


ALTERS_ITS_ARTIFACT = """
import hashlib, json, sys
out, artifact = sys.argv[1], sys.argv[2]
open(artifact, "wb").write(b"genuine")
digest = hashlib.sha256(b"genuine").hexdigest()
open(artifact, "wb").write(b"forged")  # what an agent writing to the cache would do
json.dump({"format": "qcal.executor_result", "version": 1, "status": "ok",
           "failure_kind": None, "error": None, "metrics": {"AP": 1.0}, "environment": {},
           "artifacts": [{"path": artifact, "kind": "predictions", "sha256": digest}]},
          open(out, "w"))
"""


def test_a_record_never_vouches_for_bytes_the_program_did_not_write(tmp_path: Path) -> None:
    """Gap (wave-2 review B1): the launcher hashed artifacts only when it wrote the record,
    so bytes swapped in after the program wrote them were recorded as the run's own."""
    program = tmp_path / "alters.py"
    program.write_text(ALTERS_ITS_ARTIFACT, encoding="utf-8")
    root = tmp_path / "project"
    root.mkdir()
    artifact = root / "runs" / "cache" / "p.jsonl"
    artifact.parent.mkdir(parents=True)
    command = json.dumps([sys.executable, str(program), "{result_path}", str(artifact)])
    (root / "qcal.toml").write_text(f"[executor]\ncommand = {command}\n", encoding="utf-8")
    write(root, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    record = _runner(root).run("C-a", 0)
    assert record.status == "failed"
    assert record.metrics == {}
    assert "changed after the program wrote or read it" in (record.error or "")
    assert all(a.path != "runs/cache/p.jsonl" for a in record.artifacts)
