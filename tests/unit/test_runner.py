"""Runner: single runs, refusals, provenance, planning and batches."""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from qcal.config import Config
from qcal.registry.environment import launcher_identity
from qcal.registry.executor import ExecutionResult, RunSpec, SubprocessExecutor
from qcal.registry.experiments import Experiments, ExperimentsError, load_experiments
from qcal.registry.gates import config_inputs, digest_of
from qcal.registry.records import SCHEMA_VERSION, ArtifactRef, effective
from qcal.registry.runner import (
    LauncherSignal,
    PlannedRun,
    Runner,
    RunRefusedError,
    merge_environment,
)
from qcal.registry.store import MARKER_SUFFIX, RegistryStore, pair_lock_name
from tests.conftest import (
    FakeExecutor,
    executor_command,
    experiment_script,
    experiments_yaml,
    make_record,
    run_git,
    write,
)

pytestmark = pytest.mark.rule("C7")

T0 = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)
CELLS: list[dict[str, Any]] = [
    {"id": "C-a", "detector": "atss", "precision": "fp32"},
    {"id": "C-b", "detector": "detr", "precision": "int8"},
    {"id": "D-c", "detector": "yolox", "precision": "fp16", "seeds": [5]},
]
FAILED = ExecutionResult(1, error="exit code 1; see log")


class StepClock:
    """Deterministic clock: each call advances by ``step``."""

    def __init__(self, start: datetime = T0, step: timedelta = timedelta(seconds=90)) -> None:
        self.current = start
        self.step = step

    def __call__(self) -> datetime:
        value = self.current
        self.current += self.step
        return value


class MutatingExecutor(FakeExecutor):
    """Changes the run's inputs while it executes, like a misbehaving experiment."""

    def __init__(self, action: Callable[[], object]) -> None:
        super().__init__()
        self.action = action

    def execute(self, spec: RunSpec) -> ExecutionResult:
        self.action()
        return super().execute(spec)


def counter_nonce() -> Callable[[], str]:
    counter = itertools.count()
    return lambda: f"n{next(counter)}"


def load(config: Config, cells: list[dict[str, Any]] | None = None) -> Experiments:
    write(config.root, "EXPERIMENTS.yaml", experiments_yaml(cells or CELLS, seeds=[0, 1]))
    return load_experiments(config)


def make_runner(
    config: Config,
    executor: Any,
    experiments: Experiments | None = None,
    **kwargs: Any,
) -> Runner:
    kwargs.setdefault("clock", StepClock())
    kwargs.setdefault("nonce", counter_nonce())
    kwargs.setdefault("collectors", [])
    return Runner(
        config,
        experiments or load(config),
        RegistryStore(config.path("registry_dir")),
        executor,
        **kwargs,
    )


def seed_store(config: Config, *records: Any) -> RegistryStore:
    store = RegistryStore(config.path("registry_dir"))
    for record in records:
        store.write(record)
    return store


# -- identity -----------------------------------------------------------------------------


def test_new_run_id_follows_the_configured_template(
    config: Config, fake_executor: FakeExecutor
) -> None:
    runner = make_runner(config, fake_executor, nonce=lambda: "abc123")
    assert runner.new_run_id("C-a", 0, T0) == "R20261009T120000Z-C-a-s0-abc123"


def test_run_id_template_and_timestamp_are_configurable(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config(
        """
        [registry]
        run_id_template = "{cell_id}.{seed}.{timestamp}.{nonce}"
        timestamp_format = "%Y%m%d"
        """
    )
    runner = make_runner(config, fake_executor, nonce=lambda: "x")
    assert runner.new_run_id("C-a", 3, T0) == "C-a.3.20261009.x"


@pytest.mark.parametrize(("nonce_bytes", "hex_length"), [(3, 6), (1, 2), (8, 16)])
def test_default_nonce_has_configured_length(
    make_config: Callable[[str], Config],
    fake_executor: FakeExecutor,
    nonce_bytes: int,
    hex_length: int,
) -> None:
    config = make_config(f"[registry]\nnonce_bytes = {nonce_bytes}\n")
    runner = Runner(config, load(config), RegistryStore(config.path("registry_dir")), fake_executor)
    assert re.fullmatch(f"[0-9a-f]{{{hex_length}}}", runner.nonce())


def test_default_collectors_come_from_configuration(
    config: Config, fake_executor: FakeExecutor
) -> None:
    runner = Runner(config, load(config), RegistryStore(config.path("registry_dir")), fake_executor)
    assert runner.collectors == config.str_list("registry.env_collectors")


def test_default_clock_is_timezone_aware_utc(config: Config, fake_executor: FakeExecutor) -> None:
    runner = Runner(config, load(config), RegistryStore(config.path("registry_dir")), fake_executor)
    assert runner.clock().utcoffset() == timedelta(0)


# -- config hash --------------------------------------------------------------------------


def test_config_hash_is_deterministic(config: Config, fake_executor: FakeExecutor) -> None:
    runner = make_runner(config, fake_executor)
    cell = runner.experiments.cell("C-a")
    assert runner.config_hash(cell, 0) == runner.config_hash(cell, 0)
    assert re.fullmatch(r"[0-9a-f]{64}", runner.config_hash(cell, 0))


def test_config_hash_depends_on_seed_and_cell(config: Config, fake_executor: FakeExecutor) -> None:
    runner = make_runner(config, fake_executor)
    a, b = runner.experiments.cell("C-a"), runner.experiments.cell("C-b")
    hashes = {runner.config_hash(a, 0), runner.config_hash(a, 1), runner.config_hash(b, 0)}
    assert len(hashes) == 3


def test_config_hash_changes_when_a_config_file_changes(
    config: Config, fake_executor: FakeExecutor
) -> None:
    runner = make_runner(config, fake_executor)
    cell = runner.experiments.cell("C-a")
    write(config.root, "configs/model/atss.yaml", "lr: 0.01\n")
    before = runner.config_hash(cell, 0)
    write(config.root, "configs/model/atss.yaml", "lr: 0.02\n")
    assert runner.config_hash(cell, 0) != before


def test_config_hash_ignores_files_outside_the_hash_inputs(
    config: Config, fake_executor: FakeExecutor
) -> None:
    runner = make_runner(config, fake_executor)
    cell = runner.experiments.cell("C-a")
    before = runner.config_hash(cell, 0)
    write(config.root, "notes/scratch.yaml", "irrelevant: true\n")
    write(config.root, "configs/README.md", "not a config\n")
    assert runner.config_hash(cell, 0) == before


def test_config_hash_depends_on_the_executor_command(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    first = make_config('[executor]\ncommand = ["python", "a.py"]\n')
    runner_a = make_runner(first, fake_executor)
    hash_a = runner_a.config_hash(runner_a.experiments.cell("C-a"), 0)
    second = make_config('[executor]\ncommand = ["python", "b.py"]\n')
    runner_b = make_runner(second, fake_executor)
    assert runner_b.config_hash(runner_b.experiments.cell("C-a"), 0) != hash_a


# -- single runs --------------------------------------------------------------------------


def test_run_records_a_successful_run(config: Config, fake_executor: FakeExecutor) -> None:
    runner = make_runner(config, fake_executor, nonce=lambda: "abc123")
    record = runner.run("C-a", 1)
    assert record.run_id == "R20261009T120000Z-C-a-s1-abc123"
    assert (record.cell_id, record.seed, record.seed_role, record.status) == (
        "C-a",
        1,
        "calibrator_fit_draw",
        "ok",
    )
    assert dict(record.factors) == {"detector": "atss", "precision": "fp32"}
    assert dict(record.metrics) == {"LaECE0": 10.0, "AP": 40.0}
    assert record.error is None


def test_run_writes_the_record_to_the_store(config: Config, fake_executor: FakeExecutor) -> None:
    runner = make_runner(config, fake_executor)
    record = runner.run("C-a", 0)
    assert runner.store.read(record.run_id) == record


def test_run_records_timing_from_the_clock(config: Config, fake_executor: FakeExecutor) -> None:
    record = make_runner(config, fake_executor).run("C-a", 0)
    assert (record.started_at, record.finished_at, record.duration_s) == (
        "2026-10-09T12:00:00+00:00",
        "2026-10-09T12:01:30+00:00",
        90.0,
    )


def test_run_rounds_duration_to_milliseconds(config: Config, fake_executor: FakeExecutor) -> None:
    clock = StepClock(step=timedelta(microseconds=1_234_567))
    assert make_runner(config, fake_executor, clock=clock).run("C-a", 0).duration_s == 1.235


def test_run_records_provenance(
    config: Config, fake_executor: FakeExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = config.str_list("registry.recorded_env_vars")
    for name in recorded:  # the test's environment, not the machine's (CI sets some)
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib")
    runner = make_runner(config, fake_executor)
    record = runner.run("C-a", 0)
    policy = config.root / "qcal.toml"
    assert dict(record.provenance) == {
        "git_sha": None,
        "git_dirty": None,
        "config_hash": runner.config_hash(runner.experiments.cell("C-a"), 0),
        "config_inputs_sha256": digest_of(config_inputs(config)),
        "config_sources": list(config.sources),
        "config_environment": {},
        "policy_source": "worktree",
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "experiments_sha256": runner.experiments.sha256,
        "executor": "FakeExecutor",
        "launcher": launcher_identity(config.root),
        "environment_variables": {**dict.fromkeys(recorded), "LD_LIBRARY_PATH": "/opt/lib"},
        "batch_id": None,
        "retry_of": None,
        "inputs_unverified": ["EXPERIMENTS.yaml", "qcal.toml"],  # FakeExecutor reports none
    }


def test_records_carry_the_code_schema_version(config: Config, fake_executor: FakeExecutor) -> None:
    assert make_runner(config, fake_executor).run("C-a", 0).schema_version == SCHEMA_VERSION


def test_the_removed_schema_version_key_refuses_the_run(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config("[registry]\nschema_version = 7\n")
    with pytest.raises(RunRefusedError, match=r"registry\.schema_version was removed"):
        make_runner(config, fake_executor).run("C-a", 0)
    assert fake_executor.specs == []


def test_run_hands_the_executor_a_complete_spec(
    config: Config, fake_executor: FakeExecutor
) -> None:
    record = make_runner(config, fake_executor).run("C-b", 1)
    (spec,) = fake_executor.specs
    assert spec.run_id == record.run_id
    assert (spec.cell.id, spec.seed, spec.seed_role, spec.root) == (
        "C-b",
        1,
        "calibrator_fit_draw",
        config.root,
    )
    assert spec.result_path == config.root / "runs" / "results" / f"{record.run_id}.json"
    assert spec.log_path == config.root / "runs" / "logs" / f"{record.run_id}.log"


def test_run_stores_the_log_path_relative_to_the_root(
    config: Config, fake_executor: FakeExecutor
) -> None:
    record = make_runner(config, fake_executor).run("C-a", 0)
    assert record.log_path == f"runs/logs/{record.run_id}.log"


def test_run_keeps_log_paths_outside_the_root_absolute(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor, tmp_path: Path
) -> None:
    outside = tmp_path / "shared-logs"
    config = make_config(f'[paths]\nlogs_dir = "{outside.as_posix()}"\n')
    record = make_runner(config, fake_executor).run("C-a", 0)
    assert record.log_path == str(outside / f"{record.run_id}.log")


def test_run_uses_configured_result_and_log_file_names(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config(
        '[executor]\nresult_filename = "{run_id}/result.json"\nlog_filename = "{run_id}.txt"\n'
    )
    record = make_runner(config, fake_executor).run("C-a", 0)
    (spec,) = fake_executor.specs
    assert spec.result_path.relative_to(config.root).as_posix() == (
        f"runs/results/{record.run_id}/result.json"
    )
    assert spec.log_path.name == f"{record.run_id}.txt"


def test_run_records_artifacts_of_successful_runs(config: Config) -> None:
    artifact = ArtifactRef("runs/results/x.engine", "h" * 64, "engine", 10)
    executor = FakeExecutor([ExecutionResult(0, metrics={"AP": 1.0}, artifacts=[artifact])])
    assert make_runner(config, executor).run("C-a", 0).artifacts == (artifact,)


def test_run_records_failures_without_metrics_but_with_partial_artifacts(config: Config) -> None:
    artifact = ArtifactRef("x", "h")
    result = ExecutionResult(
        0, metrics={"AP": 1.0}, artifacts=[artifact], error="bad result", failure_kind="cuda_oom"
    )
    runner = make_runner(config, FakeExecutor([result]))
    record = runner.run("C-a", 0)
    assert (record.status, record.error, dict(record.metrics), record.artifacts) == (
        "failed",
        "bad result",
        {},
        (artifact,),
    )
    assert record.failure_kind == "cuda_oom"
    assert runner.store.exists(record.run_id)


def test_a_failure_without_a_kind_is_recorded_as_unknown(config: Config) -> None:
    record = make_runner(config, FakeExecutor([FAILED])).run("C-a", 0)
    assert record.failure_kind == "unknown"


def test_an_ok_run_has_no_failure_kind(config: Config, fake_executor: FakeExecutor) -> None:
    record = make_runner(config, fake_executor).run("C-a", 0)
    assert record.failure_kind is None
    assert "failure_kind" not in record.to_dict()


class InterruptingExecutor(FakeExecutor):
    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self.exc = exc

    def execute(self, spec: RunSpec) -> ExecutionResult:
        raise self.exc


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(3)], ids=["ctrl-c", "exit"])
def test_an_interrupted_run_is_recorded_then_re_raised(config: Config, exc: BaseException) -> None:
    runner = make_runner(config, InterruptingExecutor(exc))
    with pytest.raises(type(exc)):
        runner.run("C-a", 0)
    (record,) = runner.store.load_all()
    assert (record.status, record.failure_kind) == ("failed", "interrupted")
    assert record.error == f"interrupted by {type(exc).__name__}"


def test_a_crashing_executor_is_recorded_as_unknown(config: Config) -> None:
    runner = make_runner(config, InterruptingExecutor(RuntimeError("boom")))
    record = runner.run("C-a", 0)
    assert (record.failure_kind, record.error) == ("unknown", "executor raised RuntimeError: boom")


def test_run_logs_failures(config: Config, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="qcal"):
        record = make_runner(config, FakeExecutor([FAILED])).run("C-a", 0)
    assert f"run {record.run_id} failed" in caplog.text


def test_run_uses_configured_status_names(make_config: Callable[[str], Config]) -> None:
    config = make_config('[registry]\nok_status = "done"\nfailed_status = "broken"\n')
    runner = make_runner(config, FakeExecutor([ExecutionResult(0), FAILED]))
    assert [runner.run("C-a", 0).status, runner.run("C-a", 1).status] == ["done", "broken"]


def test_run_records_the_launcher_and_the_program_under_separate_names(config: Config) -> None:
    reported = {"python": "2.7-claimed", "trt_version": "10.3.0"}
    executor = FakeExecutor([ExecutionResult(0, environment=reported)])
    record = make_runner(config, executor, collectors=["python"]).run("C-a", 0)
    assert dict(record.environment) == {
        "launcher.python": platform.python_version(),
        "launcher.python_impl": platform.python_implementation(),
        "python": "2.7-claimed",
        "trt_version": "10.3.0",
    }


@pytest.mark.parametrize(
    ("collected", "reported", "expected"),
    [
        ({"python": "3.11"}, {}, {"launcher.python": "3.11"}),
        ({}, {"trt_version": "10"}, {"trt_version": "10"}),
        ({"python": "3.11"}, {"python": "3.12"}, {"launcher.python": "3.11", "python": "3.12"}),
        (
            {"python": "3.11"},
            {"launcher.python": "2.7"},
            {"launcher.python": "3.11", "reported.launcher.python": "2.7"},
        ),
        ({}, {"reported.x": 1}, {"reported.reported.x": 1}),
        ({"gpu": None}, {"gpu": "A100"}, {"launcher.gpu": None, "gpu": "A100"}),
    ],
    ids=[
        "nothing-reported",
        "only-reported",
        "both-kept",
        "forged-launcher-key",
        "forged-reported-key",
        "none-and-value",
    ],
)
def test_merge_environment_never_lets_the_program_pass_for_the_launcher(
    collected: dict[str, Any], reported: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert merge_environment(collected, reported) == expected


def test_merge_environment_warns_about_reserved_keys(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="qcal"):
        merge_environment({"python": "3.11"}, {"launcher.python": "2.7"})
    assert "launcher.python inside a reserved namespace" in caplog.text


def test_merge_environment_is_silent_without_conflicts(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="qcal"):
        merge_environment({"python": "3.11"}, {"python": "3.11", "trt_version": "10"})
    assert caplog.records == []


def test_merge_environment_does_not_mutate_its_inputs() -> None:
    collected, reported = {"python": "3.11"}, {"python": "2.7", "x": 1}
    merge_environment(collected, reported)
    assert (collected, reported) == ({"python": "3.11"}, {"python": "2.7", "x": 1})


def test_run_records_supersession(config: Config, fake_executor: FakeExecutor) -> None:
    runner = make_runner(config, fake_executor)
    first = runner.run("C-a", 0)
    assert runner.run("C-a", 0, supersedes=first.run_id, reason="x").supersedes == first.run_id


def test_failed_rerun_does_not_supersede_the_previous_run(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    runner = make_runner(config, FakeExecutor([ExecutionResult(0, metrics={"AP": 1.0}), FAILED]))
    first = runner.run("C-a", 0)
    with caplog.at_level(logging.WARNING, logger="qcal"):
        record = runner.run("C-a", 0, supersedes=first.run_id, reason="x")
    assert (record.status, record.supersedes) == ("failed", None)
    assert f"does not supersede {first.run_id}" in caplog.text


def test_run_captures_the_config_hash_before_execution(config: Config) -> None:
    write(config.root, "configs/model.yaml", "lr: 0.01\n")
    executor = MutatingExecutor(lambda: write(config.root, "configs/model.yaml", "lr: 0.9\n"))
    runner = make_runner(config, executor)
    before = runner.config_hash(runner.experiments.cell("C-a"), 0)
    record = runner.run("C-a", 0)
    assert record.provenance["config_hash"] == before
    assert runner.config_hash(runner.experiments.cell("C-a"), 0) != before


def test_run_of_unregistered_cell_is_refused_before_execution(
    config: Config, fake_executor: FakeExecutor
) -> None:
    with pytest.raises(ExperimentsError, match="'C-z' is not pre-registered"):
        make_runner(config, fake_executor).run("C-z", 0)
    assert fake_executor.specs == []


@pytest.mark.parametrize(("cell", "seed"), [("C-a", 2), ("C-a", -1), ("D-c", 0), ("D-c", 1)])
def test_run_of_unregistered_seed_is_refused_before_execution(
    config: Config, fake_executor: FakeExecutor, cell: str, seed: int
) -> None:
    runner = make_runner(config, fake_executor)
    with pytest.raises(RunRefusedError, match=f"seed {seed} is not pre-registered for {cell}"):
        runner.run(cell, seed)
    assert fake_executor.specs == []
    assert runner.store.load_all() == []


def test_run_accepts_cell_specific_seeds(config: Config, fake_executor: FakeExecutor) -> None:
    assert make_runner(config, fake_executor).run("D-c", 5).seed == 5


@pytest.mark.parametrize("cell_id", ["bad id", "a/b", "C:1"])
def test_run_refuses_cells_whose_id_cannot_form_a_run_id(
    config: Config, fake_executor: FakeExecutor, cell_id: str
) -> None:
    runner = make_runner(config, fake_executor, experiments=load(config, [{"id": cell_id}]))
    with pytest.raises(RunRefusedError, match="cannot run"):
        runner.run(cell_id, 0)
    assert fake_executor.specs == []


# -- git provenance and clean-tree policy -------------------------------------------------


@pytest.mark.integration
def test_run_in_a_clean_repository_records_head_and_clean_tree(
    git_repo: Path, config: Config, fake_executor: FakeExecutor
) -> None:
    experiments = load(config)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "pre-registration")
    record = make_runner(config, fake_executor, experiments=experiments).run("C-a", 0)
    assert record.provenance["git_sha"] == run_git(git_repo, "rev-parse", "HEAD")
    assert record.provenance["git_dirty"] is False


@pytest.mark.integration
def test_run_in_a_dirty_repository_records_it_and_warns(
    git_repo: Path, config: Config, fake_executor: FakeExecutor, caplog: pytest.LogCaptureFixture
) -> None:
    runner = make_runner(config, fake_executor)
    run_git(git_repo, "add", "EXPERIMENTS.yaml")
    run_git(git_repo, "commit", "-q", "-m", "pre-register")
    write(git_repo, "notes.txt", "uncommitted\n")  # dirty, but not an input the gates check
    with caplog.at_level(logging.WARNING, logger="qcal"):
        record = runner.run("C-a", 0)
    assert record.provenance["git_dirty"] is True
    assert "working tree is dirty" in caplog.text


@pytest.mark.integration
def test_run_refuses_a_dirty_tree_when_a_clean_one_is_required(
    git_repo: Path, make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config("[registry]\nrequire_clean_tree = true\n")
    runner = make_runner(config, fake_executor)
    run_git(git_repo, "add", "qcal.toml", "EXPERIMENTS.yaml")
    run_git(git_repo, "commit", "-q", "-m", "require a clean tree")
    write(git_repo, "notes.txt", "uncommitted\n")
    with pytest.raises(RunRefusedError, match="working tree is dirty"):
        runner.run("C-a", 0)
    assert fake_executor.specs == []
    assert runner.store.load_all() == []


@pytest.mark.integration
def test_run_with_clean_tree_required_proceeds_when_clean(
    git_repo: Path, make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config("[registry]\nrequire_clean_tree = true\n")
    experiments = load(config)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "configure")
    record = make_runner(config, fake_executor, experiments=experiments).run("C-a", 0)
    assert record.status == "ok"


@pytest.mark.integration
def test_batch_with_clean_tree_required_runs_every_planned_run(
    git_repo: Path, make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config("[registry]\nrequire_clean_tree = true\n")
    experiments = load(config)
    # The repository's own policy: logs and results are ignored, records are committed.
    write(git_repo, ".gitignore", "runs/*\n!runs/registry/\n!runs/index.csv\n")
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "configure")
    batch = make_runner(config, fake_executor, experiments=experiments).run_batch("C-a")
    assert len(batch.completed) == 2


@pytest.mark.integration
def test_registry_outputs_never_count_as_a_dirty_tree(
    git_repo: Path, make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config("[registry]\nrequire_clean_tree = true\n")
    experiments = load(config)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "configure")
    for output in ("runs/index.csv", "runs/index.parquet", "runs/logs/x.log", "runs/results/x"):
        write(git_repo, output, "left over from an earlier run\n")
    batch = make_runner(config, fake_executor, experiments=experiments).run_batch("C-a")
    assert [r.provenance["git_dirty"] for r in batch.completed] == [False, False]


@pytest.mark.integration
def test_configured_output_locations_are_the_ones_excluded(
    git_repo: Path, make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config(
        """
        [registry]
        require_clean_tree = true

        [paths]
        registry_dir = "records"
        logs_dir = "scratch/logs"
        results_dir = "scratch/results"
        """
    )
    experiments = load(config)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "configure")
    runner = make_runner(config, fake_executor, experiments=experiments)
    runner.run("C-a", 0)
    write(git_repo, "runs/registry/stray.json", "{}\n")  # no longer a registry output
    with pytest.raises(RunRefusedError, match="working tree is dirty"):
        runner.run("C-a", 1)


@pytest.mark.integration
def test_run_records_the_head_from_before_execution(git_repo: Path, config: Config) -> None:
    experiments = load(config)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "pre-registration")
    before = run_git(git_repo, "rev-parse", "HEAD")

    def commit_during_run() -> None:
        write(git_repo, "configs/late.yaml", "x: 1\n")
        run_git(git_repo, "add", "-A")
        run_git(git_repo, "commit", "-q", "-m", "changed while running")

    runner = make_runner(config, MutatingExecutor(commit_during_run), experiments=experiments)
    record = runner.run("C-a", 0)
    assert record.provenance["git_sha"] == before
    assert run_git(git_repo, "rev-parse", "HEAD") != before


# -- planning -----------------------------------------------------------------------------


def test_plan_expands_matching_cells_and_their_seeds(
    config: Config, fake_executor: FakeExecutor
) -> None:
    planned, skipped = make_runner(config, fake_executor).plan("*")
    assert [(p.cell.id, p.seed, p.supersedes) for p in planned] == [
        ("C-a", 0, None),
        ("C-a", 1, None),
        ("C-b", 0, None),
        ("C-b", 1, None),
        ("D-c", 5, None),
    ]
    assert skipped == []


def test_plan_rejects_patterns_matching_no_cell(
    config: Config, fake_executor: FakeExecutor
) -> None:
    with pytest.raises(ExperimentsError, match="no pre-registered cell matches 'X-\\*'"):
        make_runner(config, fake_executor).plan("X-*")


def test_plan_restricts_to_requested_seeds(config: Config, fake_executor: FakeExecutor) -> None:
    planned, _ = make_runner(config, fake_executor).plan("C-*", seeds=[1])
    assert [(p.cell.id, p.seed) for p in planned] == [("C-a", 1), ("C-b", 1)]


def test_plan_refuses_requested_seeds_that_are_not_registered(
    config: Config, fake_executor: FakeExecutor
) -> None:
    with pytest.raises(RunRefusedError, match="seed 5 is not pre-registered for C-a"):
        make_runner(config, fake_executor).plan("C-a,D-c", seeds=[5])


def test_plan_skips_completed_pairs(config: Config, fake_executor: FakeExecutor) -> None:
    seed_store(config, make_record("R1", cell_id="C-a", seed=0))
    planned, skipped = make_runner(config, fake_executor).plan("C-a")
    assert [(p.cell.id, p.seed) for p in planned] == [("C-a", 1)]
    assert skipped == [("C-a", 0, "already completed as R1")]


def test_plan_retries_pairs_that_only_failed(config: Config, fake_executor: FakeExecutor) -> None:
    seed_store(config, make_record("R1", cell_id="C-a", seed=0, status="failed", metrics={}))
    planned, skipped = make_runner(config, fake_executor).plan("C-a")
    assert [(p.cell.id, p.seed, p.supersedes) for p in planned] == [
        ("C-a", 0, None),
        ("C-a", 1, None),
    ]
    assert skipped == []


def test_plan_ignores_superseded_completions(config: Config, fake_executor: FakeExecutor) -> None:
    seed_store(
        config,
        make_record("R1", cell_id="C-a", seed=0),
        make_record("R2", cell_id="C-a", seed=0, status="failed", supersedes="R1", metrics={}),
    )
    planned, _ = make_runner(config, fake_executor).plan("C-a", seeds=[0])
    assert planned == [PlannedRun(planned[0].cell, 0, None)]


def test_plan_with_rerun_supersedes_completed_pairs(
    config: Config, fake_executor: FakeExecutor
) -> None:
    seed_store(config, make_record("R1", cell_id="C-a", seed=0))
    planned, skipped = make_runner(config, fake_executor).plan("C-a", rerun=True)
    assert [(p.seed, p.supersedes) for p in planned] == [(0, "R1"), (1, None)]
    assert skipped == []


def test_plan_honours_configured_ok_status(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config('[registry]\nok_status = "done"\n')
    seed_store(config, make_record("R1", cell_id="C-a", seed=0, status="ok"))
    planned, _ = make_runner(config, fake_executor).plan("C-a", seeds=[0])
    assert len(planned) == 1


# -- batches ------------------------------------------------------------------------------


def test_run_batch_runs_every_planned_pair(config: Config, fake_executor: FakeExecutor) -> None:
    batch = make_runner(config, fake_executor).run_batch("C-*")
    assert [(r.cell_id, r.seed) for r in batch.completed] == [
        ("C-a", 0),
        ("C-a", 1),
        ("C-b", 0),
        ("C-b", 1),
    ]
    assert (batch.failed, batch.ok) == ([], True)
    assert len(fake_executor.specs) == 4


def test_run_batch_reports_skipped_pairs(config: Config, fake_executor: FakeExecutor) -> None:
    seed_store(config, make_record("R1", cell_id="C-a", seed=0))
    batch = make_runner(config, fake_executor).run_batch("C-a")
    assert batch.skipped == [("C-a", 0, "already completed as R1")]
    assert [r.seed for r in batch.completed] == [1]


def test_run_batch_is_idempotent(config: Config, fake_executor: FakeExecutor) -> None:
    runner = make_runner(config, fake_executor)
    runner.run_batch("C-a")
    second = runner.run_batch("C-a")
    assert (second.planned, second.completed, len(second.skipped)) == ([], [], 2)
    assert len(fake_executor.specs) == 2


def test_run_batch_refuses_more_runs_than_max_runs(
    config: Config, fake_executor: FakeExecutor
) -> None:
    runner = make_runner(config, fake_executor)
    with pytest.raises(RunRefusedError, match="would launch 5 runs, above the limit of 4"):
        runner.run_batch("*", max_runs=4)
    assert fake_executor.specs == []
    assert runner.store.load_all() == []


def test_run_batch_allows_exactly_max_runs(config: Config, fake_executor: FakeExecutor) -> None:
    assert len(make_runner(config, fake_executor).run_batch("*", max_runs=5).completed) == 5


def test_run_batch_with_zero_max_runs_refuses_any_run(
    config: Config, fake_executor: FakeExecutor
) -> None:
    with pytest.raises(RunRefusedError, match="above the limit of 0"):
        make_runner(config, fake_executor).run_batch("C-a", max_runs=0)


def test_run_batch_limit_defaults_to_configuration(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config("[registry]\nmax_runs_per_batch = 3\n")
    with pytest.raises(RunRefusedError, match="above the limit of 3"):
        make_runner(config, fake_executor).run_batch("*")


def test_explicit_max_runs_overrides_configuration(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config("[registry]\nmax_runs_per_batch = 1\n")
    assert len(make_runner(config, fake_executor).run_batch("*", max_runs=5).completed) == 5


def test_run_batch_limit_counts_only_planned_runs(
    make_config: Callable[[str], Config], fake_executor: FakeExecutor
) -> None:
    config = make_config("[registry]\nmax_runs_per_batch = 1\n")
    seed_store(config, make_record("R1", cell_id="C-a", seed=0))
    assert len(make_runner(config, fake_executor).run_batch("C-a").completed) == 1


def test_dry_run_plans_without_executing(config: Config, fake_executor: FakeExecutor) -> None:
    runner = make_runner(config, fake_executor)
    batch = runner.run_batch("C-*", dry_run=True)
    assert [(p.cell.id, p.seed) for p in batch.planned] == [
        ("C-a", 0),
        ("C-a", 1),
        ("C-b", 0),
        ("C-b", 1),
    ]
    assert (batch.completed, batch.failed) == ([], [])
    assert fake_executor.specs == []
    assert runner.store.load_all() == []


def test_dry_run_still_enforces_the_run_limit(config: Config, fake_executor: FakeExecutor) -> None:
    with pytest.raises(RunRefusedError, match="above the limit of 1"):
        make_runner(config, fake_executor).run_batch("*", max_runs=1, dry_run=True)


def test_run_batch_stops_at_the_first_failure(config: Config) -> None:
    executor = FakeExecutor([ExecutionResult(0, metrics={"AP": 1.0}), FAILED])
    batch = make_runner(config, executor).run_batch("C-*")
    assert [(r.cell_id, r.seed) for r in batch.completed] == [("C-a", 0)]
    assert [(r.cell_id, r.seed) for r in batch.failed] == [("C-a", 1)]
    assert batch.ok is False
    assert len(executor.specs) == 2


def test_run_batch_with_keep_going_runs_past_failures(config: Config) -> None:
    executor = FakeExecutor([FAILED, ExecutionResult(0, metrics={"AP": 1.0}), FAILED])
    batch = make_runner(config, executor).run_batch("C-*", keep_going=True)
    assert [(r.cell_id, r.seed) for r in batch.failed] == [("C-a", 0), ("C-b", 0)]
    assert [(r.cell_id, r.seed) for r in batch.completed] == [("C-a", 1), ("C-b", 1)]
    assert batch.ok is False
    assert len(executor.specs) == 4


def test_run_batch_with_rerun_supersedes_previous_records(
    config: Config, fake_executor: FakeExecutor
) -> None:
    runner = make_runner(config, fake_executor)
    first = runner.run_batch("C-a")
    second = runner.run_batch("C-a", rerun=True, reason="x")
    assert [r.supersedes for r in second.completed] == [r.run_id for r in first.completed]
    current = effective(runner.store.load_all())
    assert {r.run_id for r in current} == {r.run_id for r in second.completed}


def test_failed_rerun_in_a_batch_keeps_the_previous_result_effective(config: Config) -> None:
    runner = make_runner(config, FakeExecutor())
    (original,) = runner.run_batch("C-a", seeds=[0]).completed
    runner.executor = FakeExecutor([FAILED])
    (failed,) = runner.run_batch("C-a", seeds=[0], rerun=True, reason="x").failed
    assert failed.supersedes is None
    assert original in effective(runner.store.load_all())
    _, skipped = runner.plan("C-a", seeds=[0])
    assert skipped == [("C-a", 0, f"already completed as {original.run_id}")]


# -- end to end with a real subprocess ----------------------------------------------------


@pytest.mark.integration
def test_run_with_subprocess_executor_records_the_experiment_result(
    make_config: Callable[[str], Config], tmp_path: Path
) -> None:
    command = executor_command(experiment_script(tmp_path))
    config = make_config(f"[executor]\ncommand = {command!r}\n".replace("'", '"'))
    runner = make_runner(config, SubprocessExecutor.from_config(config))
    record = runner.run("C-b", 1)
    assert record.status == "ok"
    assert set(record.metrics) == {"LaECE0", "AP"}
    assert record.environment["trt_version"] == "test"
    assert record.provenance["executor"] == "SubprocessExecutor"
    assert (config.root / record.log_path).is_file()
    assert sys.executable in str(config.get("executor.command"))


# -- bookkeeping and concurrency (PR-A2) --------------------------------------------------


def test_a_retry_names_the_latest_failed_run_of_its_pair(config: Config) -> None:
    runner = make_runner(config, FakeExecutor([FAILED, FAILED, ExecutionResult(0)]))
    first = runner.run("C-a", 0)
    second = runner.run("C-a", 0)
    third = runner.run("C-a", 0)
    assert first.provenance["retry_of"] is None
    assert second.provenance["retry_of"] == first.run_id
    assert third.provenance["retry_of"] == second.run_id


def test_a_batch_shares_one_batch_id(config: Config, fake_executor: FakeExecutor) -> None:
    runner = make_runner(config, fake_executor)
    batch = runner.run_batch("C-*")
    ids = {r.provenance["batch_id"] for r in batch.completed}
    assert ids == {batch.batch_id}
    assert batch.batch_id is not None
    assert re.fullmatch(r"B\d{8}T\d{6}Z-n\d+", batch.batch_id)
    assert runner.run("D-c", 5).provenance["batch_id"] is None


def test_a_dry_run_batch_has_no_batch_id(config: Config, fake_executor: FakeExecutor) -> None:
    assert make_runner(config, fake_executor).run_batch("C-*", dry_run=True).batch_id is None


def test_a_pair_locked_by_another_launcher_is_refused(
    config: Config, fake_executor: FakeExecutor
) -> None:
    runner = make_runner(config, fake_executor)
    directory = config.path("inflight_dir")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / pair_lock_name("C-a", 0)).open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with pytest.raises(RunRefusedError, match="being run by another launcher"):
            runner.run("C-a", 0)
    assert fake_executor.specs == []
    assert runner.run("C-a", 0).status == "ok"  # the lock is released with its holder


def test_a_running_run_has_an_inflight_marker_that_is_removed_after(config: Config) -> None:
    seen: list[list[str]] = []

    def look() -> None:
        seen.append(sorted(p.name for p in config.path("inflight_dir").glob(f"*{MARKER_SUFFIX}")))

    record = make_runner(config, MutatingExecutor(look)).run("C-a", 0)
    assert seen == [[f"{record.run_id}{MARKER_SUFFIX}"]]
    assert list(config.path("inflight_dir").glob(f"*{MARKER_SUFFIX}")) == []


def test_the_marker_is_removed_when_the_run_is_interrupted(config: Config) -> None:
    class Interrupt(FakeExecutor):
        def execute(self, spec: RunSpec) -> ExecutionResult:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        make_runner(config, Interrupt()).run("C-a", 0)
    assert list(config.path("inflight_dir").glob(f"*{MARKER_SUFFIX}")) == []


def test_lock_names_are_safe_for_any_cell_id() -> None:
    name = pair_lock_name("a/b c", 3)
    assert "/" not in name
    assert " " not in name
    assert name.startswith("a_b_c@3-")
    assert name != pair_lock_name("a_b_c", 3)  # the digest keeps sanitised names apart


LOCK_HOLDER = """
import fcntl, sys, time
handle = open(sys.argv[1], "a")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep(60)
"""


@pytest.mark.integration
def test_a_second_launcher_process_is_refused_while_the_first_holds_the_pair(
    config: Config, fake_executor: FakeExecutor, tmp_path: Path
) -> None:
    directory = config.path("inflight_dir")
    directory.mkdir(parents=True, exist_ok=True)
    holder = tmp_path / "holder.py"
    holder.write_text(LOCK_HOLDER)
    lock = directory / pair_lock_name("C-a", 0)
    proc = subprocess.Popen(
        [sys.executable, str(holder), str(lock)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(RunRefusedError, match="being run by another launcher"):
            make_runner(config, fake_executor).run("C-a", 0)
    finally:
        proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
    assert make_runner(config, fake_executor).run("C-a", 0).status == "ok"


def test_a_record_carries_what_the_run_cost(config: Config) -> None:
    result = ExecutionResult(0, resources={"wall_s": 2.0, "cache_hits": 1})
    record = make_runner(config, FakeExecutor([result])).run("C-a", 0)
    assert dict(record.resources) == {"wall_s": 2.0, "cache_hits": 1}


# -- signals around a run (wave-2 review, B5 and B6) ------------------------------------------


class SignallingExecutor(FakeExecutor):
    """Sends the launcher a signal while the program "runs", as `kill` or a closed terminal."""

    def __init__(self, number: int) -> None:
        super().__init__()
        self.number = number

    def execute(self, spec: RunSpec) -> ExecutionResult:
        os.kill(os.getpid(), self.number)
        time.sleep(5)  # the handler interrupts this
        raise AssertionError("the signal did not interrupt the run")


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGHUP], ids=["sigterm", "sighup"])
def test_sigterm_and_sighup_end_a_run_like_ctrl_c(config: Config, number: int) -> None:
    previous = signal.getsignal(number)
    runner = make_runner(config, SignallingExecutor(number))
    with pytest.raises(LauncherSignal):
        runner.run("C-a", 0)
    (record,) = runner.store.load_all()
    assert (record.status, record.failure_kind) == ("failed", "interrupted")
    assert signal.getsignal(number) is previous  # the launcher's own handler is restored


def test_an_interrupt_while_recording_waits_for_the_record(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C after the program finished (while collectors import torch) used to lose it."""
    from qcal.registry import runner as runner_module

    def collect_then_interrupt(*_: Any, **__: Any) -> dict[str, Any]:
        os.kill(os.getpid(), signal.SIGINT)
        return {}

    monkeypatch.setattr(runner_module, "collect_environment", collect_then_interrupt)
    runner = make_runner(config, FakeExecutor())
    # Another live thread, as a launcher may have: the kernel can deliver the signal to it,
    # so masking the main thread alone would not hold it.
    done = threading.Event()
    other = threading.Thread(target=done.wait, daemon=True)
    other.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run("C-a", 0)
    finally:
        done.set()
        other.join()
    (record,) = runner.store.load_all()
    assert record.status == "ok"  # the run finished; the record says so
    assert list(config.path("inflight_dir").glob(f"*{MARKER_SUFFIX}")) == []


def test_the_marker_stays_when_the_record_cannot_be_written(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit reports a run that left no record; deleting its marker would hide it."""
    runner = make_runner(config, FakeExecutor())

    def full_disk(_: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(runner.store, "write", full_disk)
    with pytest.raises(OSError, match="no space"):
        runner.run("C-a", 0)
    assert len(list(config.path("inflight_dir").glob(f"*{MARKER_SUFFIX}"))) == 1
