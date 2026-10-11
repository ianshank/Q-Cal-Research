"""Executors: placeholder validation, the result-file contract and real subprocess runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qcal.components import UnknownComponentError
from qcal.config import Config, ConfigError
from qcal.registry import executor as executor_module
from qcal.registry.executor import (
    EXECUTORS,
    PLACEHOLDERS,
    ExecutionResult,
    Executor,
    ExecutorNotConfiguredError,
    RunSpec,
    SubprocessExecutor,
    build_executor,
    read_result,
    sha256_file,
    validate_template,
)
from qcal.registry.experiments import Cell
from qcal.registry.records import ArtifactRef
from tests.conftest import FakeExecutor, executor_command, experiment_script, write

CELL = Cell("C-a", {"detector": "atss"})


def make_spec(root: Path, run_id: str = "R1", seed: int = 0) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        cell=CELL,
        seed=seed,
        seed_role="calibrator_fit_draw",
        result_path=root / "runs" / "results" / f"{run_id}.json",
        log_path=root / "runs" / "logs" / f"{run_id}.log",
        root=root,
    )


def make_executor(
    root: Path, command: list[str], timeout_s: float | None = None
) -> SubprocessExecutor:
    return SubprocessExecutor(command, timeout_s=timeout_s, env_prefix="QCAL_RUN_", root=root)


def script(tmp_path: Path, body: str, name: str = "program.py") -> Path:
    return write(tmp_path, name, body)


def result_file(root: Path, payload: Any) -> Path:
    path = root / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), "utf-8")
    return path


# -- templates ----------------------------------------------------------------------------


def test_placeholders_are_the_documented_set() -> None:
    documented = {
        "python", "run_id", "cell_id", "seed", "seed_role", "result_path", "log_path", "root"
    }  # fmt: skip
    assert documented == PLACEHOLDERS


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["python", "train.py"],
        [f"{{{name}}}" for name in sorted(PLACEHOLDERS)],
        ["--out={result_path}", "--seed={seed:>4}", "{seed!r}"],
        ["{{literal braces}}"],
        ["--seed={seed:>{seed}}"],
    ],
)
def test_validate_template_accepts_documented_placeholders(command: list[str]) -> None:
    validate_template(command)


@pytest.mark.parametrize(
    ("part", "name"),
    [
        ("{cell}", "cell"),
        ("--lr={lr}", "lr"),
        ("{}", ""),
        ("{0}", "0"),
        ("{root.parent}", "root.parent"),
        ("{env[HOME]}", "env[HOME]"),
    ],
)
def test_validate_template_rejects_other_fields(part: str, name: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(f"unknown placeholder {{{name}}}")):
        validate_template(["python", part])


@pytest.mark.parametrize(
    ("part", "name"),
    [("--seed={seed:{lr}}", "lr"), ("{run_id:{seed:{deep}}}", "deep"), ("{root:>{}}", "")],
)
def test_validate_template_rejects_unknown_nested_placeholders(part: str, name: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(f"unknown placeholder {{{name}}}")):
        validate_template([part])


def test_run_spec_values_cover_every_placeholder(tmp_path: Path) -> None:
    values = make_spec(tmp_path, seed=3).values()
    assert set(values) == PLACEHOLDERS
    assert values["seed"] == "3"
    assert values["cell_id"] == "C-a"
    assert all(isinstance(v, str) for v in values.values())


@pytest.mark.parametrize(
    ("returncode", "error", "ok"),
    [(0, None, True), (1, None, False), (-1, None, False), (0, "bad result", False)],
)
def test_execution_result_ok_needs_zero_exit_and_no_error(
    returncode: int, error: str | None, ok: bool
) -> None:
    assert ExecutionResult(returncode, error=error).ok is ok


# -- construction -------------------------------------------------------------------------


def test_empty_command_means_not_configured(tmp_path: Path) -> None:
    with pytest.raises(ExecutorNotConfiguredError, match=r"executor\.command is empty"):
        make_executor(tmp_path, [])


def test_not_configured_is_a_configuration_error() -> None:
    assert issubclass(ExecutorNotConfiguredError, ConfigError)


def test_constructor_validates_the_template(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown placeholder"):
        make_executor(tmp_path, ["python", "{lr}"])


def test_argv_substitutes_placeholders(tmp_path: Path) -> None:
    executor = make_executor(tmp_path, ["prog", "{cell_id}", "--seed={seed}", "{result_path}"])
    spec = make_spec(tmp_path, seed=2)
    assert executor.argv(spec) == ["prog", "C-a", "--seed=2", str(spec.result_path)]


def test_environment_exports_prefixed_spec_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNRELATED_TEST_VARIABLE", "kept")
    env = make_executor(tmp_path, ["prog"]).environment(make_spec(tmp_path, seed=4))
    assert env["UNRELATED_TEST_VARIABLE"] == "kept"
    assert {k: v for k, v in env.items() if k.startswith("QCAL_RUN_")} == {
        "QCAL_RUN_RUN_ID": "R1",
        "QCAL_RUN_CELL_ID": "C-a",
        "QCAL_RUN_SEED": "4",
        "QCAL_RUN_SEED_ROLE": "calibrator_fit_draw",
        "QCAL_RUN_RESULT_PATH": str(tmp_path / "runs" / "results" / "R1.json"),
        "QCAL_RUN_LOG_PATH": str(tmp_path / "runs" / "logs" / "R1.log"),
        "QCAL_RUN_ROOT": str(tmp_path),
        "QCAL_RUN_PYTHON": sys.executable,
    }


def test_from_config_reads_executor_settings(make_config: Callable[[str], Config]) -> None:
    config = make_config(
        """
        [executor]
        command = ["python", "run.py", "{cell_id}"]
        timeout_s = 2.5
        env_prefix = "X_"
        """
    )
    executor = SubprocessExecutor.from_config(config)
    assert (executor.command, executor.timeout_s, executor.env_prefix, executor.root) == (
        ["python", "run.py", "{cell_id}"],
        2.5,
        "X_",
        config.root,
    )


def test_from_config_treats_zero_timeout_as_no_timeout(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config('[executor]\ncommand = ["python"]\ntimeout_s = 0\n')
    assert SubprocessExecutor.from_config(config).timeout_s is None


def test_from_config_with_default_empty_command_is_not_configured(config: Config) -> None:
    with pytest.raises(ExecutorNotConfiguredError):
        SubprocessExecutor.from_config(config)


def test_build_executor_uses_the_subprocess_executor_by_default(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config('[executor]\ncommand = ["python", "{cell_id}"]\n')
    executor = build_executor(config)
    assert isinstance(executor, SubprocessExecutor)
    assert isinstance(executor, Executor)


def test_build_executor_rejects_unknown_names(config: Config) -> None:
    with pytest.raises(UnknownComponentError, match="unknown executor 'slurm'"):
        build_executor(config, "slurm")


def test_subprocess_executor_is_registered() -> None:
    assert "subprocess" in EXECUTORS


def test_fake_executor_satisfies_the_protocol() -> None:
    assert isinstance(FakeExecutor(), Executor)


# -- hashing ------------------------------------------------------------------------------


def test_sha256_file_hashes_across_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_module, "_HASH_CHUNK", 7)
    data = bytes(range(256)) * 3
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_file_of_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


# -- the result-file contract -------------------------------------------------------------


def test_read_result_parses_metrics_environment_and_artifacts(tmp_path: Path) -> None:
    engine = write(tmp_path, "runs/results/model.engine", "ENGINE")
    path = result_file(
        tmp_path,
        {
            "metrics": {"AP": 41, "LaECE0": 12.5},
            "artifacts": [{"path": "runs/results/model.engine", "kind": "engine"}],
            "environment": {"trt_version": "10.3.0"},
        },
    )
    result = read_result(path, tmp_path)
    assert result.ok
    assert result.metrics == {"AP": 41.0, "LaECE0": 12.5}
    assert all(type(v) is float for v in result.metrics.values())
    assert result.environment == {"trt_version": "10.3.0"}
    assert result.artifacts == [
        ArtifactRef("runs/results/model.engine", sha256_file(engine), "engine", 6)
    ]


def test_read_result_accepts_documented_metric_names(tmp_path: Path) -> None:
    names = {"AP@50": 1, "latency_p50_ms": 2, "AP.small": 3, "brier-iou": 4}
    result = read_result(result_file(tmp_path, {"metrics": names}), tmp_path)
    assert result.metrics == {k: float(v) for k, v in names.items()}


def test_read_result_passes_the_return_code_through(tmp_path: Path) -> None:
    assert read_result(result_file(tmp_path, {}), tmp_path, 3).returncode == 3


@pytest.mark.parametrize(
    "payload",
    [{}, {"artifacts": None}, {"environment": "not a mapping"}, {"metrics": {}}],
    ids=["empty", "null-artifacts", "non-mapping-environment", "empty-metrics"],
)
def test_read_result_tolerates_optional_sections(tmp_path: Path, payload: dict[str, Any]) -> None:
    result = read_result(result_file(tmp_path, payload), tmp_path)
    assert (result.ok, result.metrics, result.artifacts, result.environment) == (
        True,
        {},
        [],
        {},
    )


def test_read_result_reports_a_missing_file(tmp_path: Path) -> None:
    result = read_result(tmp_path / "nothing.json", tmp_path)
    assert result.error is not None
    assert "wrote no result file" in result.error


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{ not json", "result file is not JSON"),
        ("[1, 2]", "must contain a JSON object"),
        ({"metrics": [1.0]}, "'metrics' must be an object"),
        ({"metrics": None}, "'metrics' must be an object"),
        ({"metrics": {"AP": True}}, "non-finite or non-numeric metrics: AP"),
        ({"metrics": {"AP": "41"}}, "non-finite or non-numeric metrics: AP"),
        ('{"metrics": {"AP": NaN}}', "non-finite or non-numeric metrics: AP"),
        ('{"metrics": {"AP": Infinity, "x": 1}}', "non-finite or non-numeric metrics: AP"),
        ({"metrics": {"a:b": 1.0}}, "metric names not usable in claim references: a:b"),
        ({"metrics": {"a b": 1.0}}, "metric names not usable in claim references: a b"),
        ({"metrics": {"AP$_0$": 1.0}}, "metric names not usable in claim references: AP$_0$"),
        ({"metrics": {"-AP": 1.0}}, "metric names not usable in claim references: -AP"),
        ({"artifacts": {"path": "x"}}, "'artifacts' must be a list"),
        ({"artifacts": ["x"]}, "artifact entries need a 'path'"),
        ({"artifacts": [{"kind": "engine"}]}, "artifact entries need a 'path'"),
        ({"artifacts": [{"path": 3}]}, "artifact entries need a 'path'"),
        ({"artifacts": [{"path": "missing.engine"}]}, "artifact missing.engine does not exist"),
        ({"artifacts": [{"path": "."}]}, "artifact . does not exist"),
    ],
)
def test_read_result_turns_invalid_documents_into_errors(
    tmp_path: Path, payload: Any, message: str
) -> None:
    result = read_result(result_file(tmp_path, payload), tmp_path)
    assert result.error is not None
    assert message in result.error
    assert (result.ok, result.metrics, result.artifacts) == (False, {}, [])


def test_read_result_reports_undecodable_result_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_bytes(b"\xff\xfe\x00garbage")
    result = read_result(path, tmp_path)
    assert result.ok is False
    assert result.error is not None
    assert result.error.startswith("result file is not UTF-8:")


def test_read_result_shows_absolute_artifacts_inside_root_relative(tmp_path: Path) -> None:
    engine = write(tmp_path, "out/model.engine", "E")
    path = result_file(tmp_path, {"artifacts": [{"path": str(engine)}]})
    (artifact,) = read_result(path, tmp_path).artifacts
    assert (artifact.path, artifact.kind, artifact.size) == ("out/model.engine", "", 1)


def test_read_result_keeps_absolute_artifacts_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = write(tmp_path, "elsewhere/cache.bin", "CACHE")
    path = result_file(root, {"artifacts": [{"path": str(outside), "kind": "calib_cache"}]})
    (artifact,) = read_result(path, root).artifacts
    assert artifact == ArtifactRef(str(outside), sha256_file(outside), "calib_cache", 5)


# -- real subprocesses --------------------------------------------------------------------


@pytest.mark.integration
def test_execute_runs_the_experiment_and_reads_its_result(tmp_path: Path, config: Config) -> None:
    executor = make_executor(config.root, executor_command(experiment_script(tmp_path)))
    result = executor.execute(make_spec(config.root, seed=2))
    base = (zlib.crc32(b"C-a") % 1000) / 100.0
    assert result.ok
    assert result.metrics == {"LaECE0": round(base + 2, 4), "AP": 40.2}
    assert result.environment == {"trt_version": "test"}


@pytest.mark.integration
def test_execute_creates_log_and_result_directories(tmp_path: Path, config: Config) -> None:
    executor = make_executor(config.root, executor_command(experiment_script(tmp_path)))
    spec = make_spec(config.root)
    executor.execute(spec)
    assert spec.log_path.is_file()
    assert spec.result_path.is_file()


@pytest.mark.integration
def test_execute_passes_spec_environment_and_runs_in_root(tmp_path: Path, config: Config) -> None:
    program = script(
        tmp_path,
        """
        import json, os, sys
        env = {k: v for k, v in os.environ.items() if k.startswith("QCAL_RUN_")}
        env["cwd"] = os.getcwd()
        json.dump({"metrics": {}, "environment": env}, open(sys.argv[1], "w"))
        """,
    )
    executor = make_executor(config.root, [sys.executable, str(program), "{result_path}"])
    result = executor.execute(make_spec(config.root, seed=1))
    assert result.environment["cwd"] == os.fspath(config.root)
    assert result.environment["QCAL_RUN_SEED"] == "1"
    assert result.environment["QCAL_RUN_CELL_ID"] == "C-a"
    assert result.environment["QCAL_RUN_SEED_ROLE"] == "calibrator_fit_draw"


@pytest.mark.integration
def test_execute_captures_stdout_and_stderr_in_the_log(tmp_path: Path, config: Config) -> None:
    program = script(
        tmp_path,
        """
        import json, sys
        print("to stdout", flush=True)
        print("to stderr", file=sys.stderr, flush=True)
        json.dump({"metrics": {"AP": 1}}, open(sys.argv[1], "w"))
        """,
    )
    spec = make_spec(config.root)
    make_executor(config.root, [sys.executable, str(program), "{result_path}"]).execute(spec)
    log = spec.log_path.read_text("utf-8")
    assert "to stdout" in log
    assert "to stderr" in log


@pytest.mark.integration
def test_execute_reports_non_zero_exit(tmp_path: Path, config: Config) -> None:
    program = script(tmp_path, "import sys\nprint('oops')\nsys.exit(3)\n")
    spec = make_spec(config.root)
    result = make_executor(config.root, [sys.executable, str(program)]).execute(spec)
    assert (result.returncode, result.ok, result.metrics) == (3, False, {})
    assert result.error == f"exit code 3; see {spec.log_path}"
    assert "oops" in spec.log_path.read_text("utf-8")


@pytest.mark.integration
def test_execute_reports_missing_result_file(tmp_path: Path, config: Config) -> None:
    program = script(tmp_path, "print('forgot to write the result')\n")
    result = make_executor(config.root, [sys.executable, str(program)]).execute(
        make_spec(config.root)
    )
    assert (result.returncode, result.ok) == (0, False)
    assert result.error is not None
    assert "wrote no result file" in result.error


@pytest.mark.integration
def test_execute_reports_timeout(tmp_path: Path, config: Config) -> None:
    program = script(tmp_path, "import time\ntime.sleep(60)\n")
    executor = make_executor(config.root, [sys.executable, str(program)], timeout_s=0.25)
    result = executor.execute(make_spec(config.root))
    assert (result.returncode, result.error) == (-1, "timed out after 0.25s")


@pytest.mark.integration
def test_execute_reports_missing_program(tmp_path: Path, config: Config) -> None:
    missing = tmp_path / "no-such-program"
    result = make_executor(config.root, [str(missing), "{cell_id}"]).execute(make_spec(config.root))
    assert result.returncode == -1
    assert result.error is not None
    assert result.error.startswith(f"cannot start {missing}:")
