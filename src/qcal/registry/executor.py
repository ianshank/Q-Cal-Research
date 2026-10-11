"""How a registered run is actually executed.

The default :class:`SubprocessExecutor` runs an argv template from configuration
and reads the result envelope the experiment program writes on every exit:

    {"format": "qcal.executor_result", "version": 1,
     "status": "ok",                      # or "failed"
     "failure_kind": null,                # one of records.FAILURE_KINDS when failed
     "error": null,
     "metrics": {"AP": 41.2, ...},
     "artifacts": [{"path": "runs/results/x.engine", "kind": "engine"}],
     "environment": {"trt_version": "10.3.0"}}

The program runs in its own process group. On a timeout, or when the launcher itself is
interrupted, the whole group gets SIGTERM and, after ``executor.kill_grace_s``, SIGKILL, so
worker processes the program started cannot outlive the run. POSIX only.

Templates may use only the documented placeholders, and the CLI accepts no
free-form overrides, so a run cannot drift from the pre-registered cell. ``{python}`` is
the launcher's own interpreter, so the hashed template names no machine path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import signal
import string
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from qcal.components import ComponentRegistry
from qcal.config import Config, ConfigError, run_neutral_environment, without_config_environment
from qcal.log import get_logger
from qcal.registry.experiments import Cell
from qcal.registry.records import FAILURE_KINDS, METRIC_NAME_PATTERN, ArtifactRef

_log = get_logger("registry.executor")

PLACEHOLDERS: Final = frozenset(
    {"python", "run_id", "cell_id", "seed", "seed_role", "result_path", "log_path", "root"}
)
_HASH_CHUNK: Final = 1 << 20
RESULT_FORMAT: Final = "qcal.executor_result"
RESULT_VERSION: Final = 1
STATUS_OK: Final = "ok"
STATUS_FAILED: Final = "failed"
STATUSES: Final = (STATUS_OK, STATUS_FAILED)
#: How a failure the launcher detects itself is named; it overrides the program's.
TIMEOUT_KIND: Final = "timeout"
UNKNOWN_KIND: Final = "unknown"


class ExecutorNotConfiguredError(ConfigError):
    """No experiment command is configured yet (Phase 1 provides one)."""


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    cell: Cell
    seed: int
    seed_role: str
    result_path: Path
    log_path: Path
    root: Path
    python: str = field(default_factory=lambda: sys.executable)

    def values(self) -> dict[str, str]:
        return {
            "python": self.python,
            "run_id": self.run_id,
            "cell_id": self.cell.id,
            "seed": str(self.seed),
            "seed_role": self.seed_role,
            "result_path": str(self.result_path),
            "log_path": str(self.log_path),
            "root": str(self.root),
        }


@dataclass
class ExecutionResult:
    returncode: int
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    failure_kind: str | None = None
    #: What the process cost, from ``os.wait4`` (wall and CPU seconds, peak RSS).
    resources: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


@runtime_checkable
class Executor(Protocol):
    def execute(self, spec: RunSpec) -> ExecutionResult: ...


ExecutorFactory = Callable[[Config], Executor]
EXECUTORS: ComponentRegistry[ExecutorFactory] = ComponentRegistry("executor")


def _placeholder_names(text: str) -> list[str]:
    """Every replacement-field name in ``text``, including fields nested in format specs."""
    names: list[str] = []
    for _, name, spec, _ in string.Formatter().parse(text):
        if name is not None:
            names.append(name)
        if spec:
            names.extend(_placeholder_names(spec))
    return names


def validate_template(command: Sequence[str]) -> None:
    for part in command:
        for name in _placeholder_names(part):
            if name not in PLACEHOLDERS:
                raise ConfigError(
                    f"executor.command uses unknown placeholder {{{name}}}; "
                    f"allowed: {sorted(PLACEHOLDERS)}"
                )


def result_envelope(
    status: str,
    *,
    metrics: Mapping[str, float] | None = None,
    artifacts: Sequence[Mapping[str, str]] | None = None,
    environment: Mapping[str, Any] | None = None,
    failure_kind: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """The document an experiment program writes to ``{result_path}``, ok or failed."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    if status == STATUS_OK and (failure_kind is not None or error is not None):
        raise ValueError("an ok result has no failure_kind or error")
    if status == STATUS_FAILED and failure_kind not in FAILURE_KINDS:
        raise ValueError(f"a failed result needs a failure_kind in {FAILURE_KINDS}")
    return {
        "format": RESULT_FORMAT,
        "version": RESULT_VERSION,
        "status": status,
        "failure_kind": failure_kind,
        "error": error,
        "metrics": dict(metrics or {}),
        "artifacts": [dict(a) for a in artifacts or ()],
        "environment": dict(environment or {}),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class SubprocessExecutor:
    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_s: float | None,
        env_prefix: str,
        root: Path,
        pass_env: Mapping[str, str] | None = None,
        kill_grace_s: float = 30.0,
        poll_interval_s: float = 0.2,
    ) -> None:
        if not command:
            raise ExecutorNotConfiguredError(
                "executor.command is empty: no experiment program is configured yet. "
                "Set it in qcal.toml once Phase 1 provides one."
            )
        validate_template(command)
        self.command = list(command)
        self.timeout_s = timeout_s
        self.env_prefix = env_prefix
        self.root = root
        self.pass_env = dict(pass_env or {})
        self.kill_grace_s = kill_grace_s
        self.poll_interval_s = poll_interval_s

    @classmethod
    def from_config(cls, config: Config) -> SubprocessExecutor:
        timeout = config.float_value("executor.timeout_s")
        return cls(
            config.str_list("executor.command"),
            timeout_s=timeout if timeout > 0 else None,
            env_prefix=config.str_value("executor.env_prefix"),
            root=config.root,
            pass_env=run_neutral_environment(config),
            kill_grace_s=config.float_value("executor.kill_grace_s"),
            poll_interval_s=config.float_value("executor.poll_interval_s"),
        )

    def argv(self, spec: RunSpec) -> list[str]:
        values = spec.values()
        return [part.format_map(values) for part in self.command]

    def environment(self, spec: RunSpec) -> dict[str, str]:
        """The child's environment: configuration variables removed, run-neutral ones re-added.

        The experiment program loads configuration itself; a ``QCAL__*`` override or a
        ``QCAL_CONFIG`` inherited from the launcher's shell would change what it reads.
        """
        env = without_config_environment(os.environ)
        env.update(self.pass_env)
        env.update({f"{self.env_prefix}{k.upper()}": v for k, v in spec.values().items()})
        return env

    def execute(self, spec: RunSpec) -> ExecutionResult:
        argv = self.argv(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.result_path.parent.mkdir(parents=True, exist_ok=True)
        _log.info("run %s: %s", spec.run_id, " ".join(argv))
        started = time.monotonic()
        try:
            with spec.log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    argv,
                    cwd=self.root,
                    env=self.environment(spec),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # its own process group, killed as a whole
                )
        except OSError as exc:
            return ExecutionResult(
                -1, error=f"cannot start {argv[0]}: {exc}", failure_kind=UNKNOWN_KIND
            )
        returncode, usage, timed_out = self._wait(proc)
        resources = _resources(time.monotonic() - started, usage)
        result = read_result(spec.result_path, self.root, returncode)
        result.resources = resources
        if timed_out:
            result.returncode, result.metrics = -1, {}
            result.error = f"timed out after {self.timeout_s}s"
            result.failure_kind = TIMEOUT_KIND
        elif returncode != 0:
            detail = f": {result.error}" if result.error else ""
            result.error = f"exit code {returncode}; see {spec.log_path}{detail}"
            result.failure_kind = result.failure_kind or UNKNOWN_KIND
        return result

    # -- the process group ------------------------------------------------------------
    def _wait(self, proc: subprocess.Popen[bytes]) -> tuple[int, Any, bool]:
        """Wait for the program; returns (exit code, rusage, timed out). On a timeout or any
        exception in the launcher (Ctrl-C included) the group is terminated first."""
        deadline = None if self.timeout_s is None else time.monotonic() + self.timeout_s
        try:
            while True:
                reaped = _reap(proc, block=False)
                if reaped is not None:
                    return (*reaped, False)
                if deadline is not None and time.monotonic() >= deadline:
                    _log.error(
                        "run timed out after %ss; terminating its process group", self.timeout_s
                    )
                    return (*self._terminate(proc), True)
                time.sleep(self.poll_interval_s)
        except BaseException:
            _log.error("launcher interrupted; terminating the program's process group")
            self._terminate(proc)
            raise

    def _terminate(self, proc: subprocess.Popen[bytes]) -> tuple[int, Any]:
        """SIGTERM to the whole group, SIGKILL after the grace period; returns the leader's
        (exit code, rusage). A second interrupt while waiting goes straight to SIGKILL, so the
        program never outlives the launcher."""
        group = proc.pid  # start_new_session: the program leads its own group
        _signal_group(group, signal.SIGTERM)
        try:
            reaped = self._await_group(proc, group)
        except BaseException:
            _signal_group(group, signal.SIGKILL)
            _reap(proc, block=True)
            raise
        if reaped is not None and not _group_alive(group):
            return reaped
        _log.error(
            "process group %d outlived %ss after SIGTERM; sending SIGKILL", group, self.kill_grace_s
        )
        _signal_group(group, signal.SIGKILL)
        return reaped if reaped is not None else _reap(proc, block=True) or (-signal.SIGKILL, None)

    def _await_group(self, proc: subprocess.Popen[bytes], group: int) -> tuple[int, Any] | None:
        """Wait up to the grace period for the leader to exit and the group to empty."""
        reaped = _reap(proc, block=False)
        deadline = time.monotonic() + self.kill_grace_s
        while time.monotonic() < deadline:
            if reaped is None:
                reaped = _reap(proc, block=False)
            if reaped is not None and not _group_alive(group):
                return reaped
            time.sleep(self.poll_interval_s)
        return reaped


def _reap(proc: subprocess.Popen[bytes], *, block: bool) -> tuple[int, Any] | None:
    """The leader's (exit code, rusage) once it has exited, else ``None``."""
    if proc.returncode is not None:
        return proc.returncode, None
    try:
        pid, status, usage = os.wait4(proc.pid, 0 if block else os.WNOHANG)
    except ChildProcessError:  # already reaped elsewhere
        return proc.returncode if proc.returncode is not None else -1, None
    if pid == 0:
        return None
    proc.returncode = os.waitstatus_to_exitcode(status)
    return proc.returncode, usage


def _signal_group(group: int, signum: int) -> None:
    with contextlib.suppress(ProcessLookupError):  # the whole group has already exited
        os.killpg(group, signum)


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _resources(wall_s: float, usage: Any) -> dict[str, Any]:
    """Wall and CPU seconds and peak resident memory (KiB on Linux) of the program's leader."""
    resources: dict[str, Any] = {"wall_s": round(wall_s, 3)}
    if usage is not None:
        resources.update(
            cpu_user_s=round(usage.ru_utime, 3),
            cpu_sys_s=round(usage.ru_stime, 3),
            peak_rss_kib=int(usage.ru_maxrss),
        )
    return resources


class _ResultError(ValueError):
    pass


def _parse_result_document(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise _ResultError(f"experiment wrote no result file at {path}")
    try:
        data = json.loads(path.read_text("utf-8"))
    except UnicodeDecodeError as exc:
        raise _ResultError(f"result file is not UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _ResultError(f"result file is not JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise _ResultError("result file must contain a JSON object")
    return data


def _parse_metrics(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise _ResultError("'metrics' must be an object")
    bad = [
        k
        for k, v in raw.items()
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
    ]
    if bad:
        raise _ResultError(f"non-finite or non-numeric metrics: {', '.join(bad)}")
    unsafe = [str(k) for k in raw if not METRIC_NAME_PATTERN.fullmatch(str(k))]
    if unsafe:
        raise _ResultError(f"metric names not usable in claim references: {', '.join(unsafe)}")
    return {k: float(v) for k, v in raw.items()}


def _parse_artifacts(raw: Any, root: Path) -> list[ArtifactRef]:
    if not isinstance(raw, list):
        raise _ResultError("'artifacts' must be a list")
    artifacts: list[ArtifactRef] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise _ResultError("artifact entries need a 'path'")
        given = Path(entry["path"])
        absolute = given if given.is_absolute() else root / given
        if not absolute.is_file():
            raise _ResultError(f"artifact {entry['path']} does not exist")
        shown = (
            absolute.relative_to(root).as_posix()
            if absolute.is_relative_to(root)
            else str(absolute)
        )
        kind = str(entry.get("kind", ""))
        artifacts.append(ArtifactRef(shown, sha256_file(absolute), kind, absolute.stat().st_size))
    return artifacts


def _parse_envelope(data: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    """(status, failure_kind, error) of a well-formed envelope; anything else is refused."""
    if data.get("format") != RESULT_FORMAT or data.get("version") != RESULT_VERSION:
        raise _ResultError(
            f"result file is not a {RESULT_FORMAT} version {RESULT_VERSION} envelope "
            f"(format={data.get('format')!r}, version={data.get('version')!r})"
        )
    status, kind, error = data.get("status"), data.get("failure_kind"), data.get("error")
    if status not in STATUSES:
        raise _ResultError(f"result status must be one of {STATUSES}, got {status!r}")
    if kind is not None and kind not in FAILURE_KINDS:
        raise _ResultError(f"failure_kind must be one of {FAILURE_KINDS}, got {kind!r}")
    if error is not None and not isinstance(error, str):
        raise _ResultError("result error must be a string or null")
    return status, kind, error


def _partial_artifacts(raw: Any, root: Path) -> list[ArtifactRef]:
    """A failed run's artifacts: those that exist are kept, the rest are logged and skipped."""
    if not isinstance(raw, list):
        return []
    kept: list[ArtifactRef] = []
    for entry in raw:
        try:
            kept += _parse_artifacts([entry], root)
        except _ResultError as exc:
            _log.warning("failed run: artifact skipped: %s", exc)
    return kept


def read_result(path: Path, root: Path, returncode: int = 0) -> ExecutionResult:
    """Parse the result envelope a program wrote, after any exit.

    An ok run needs exit code 0, status ``ok`` and valid metrics and artifacts. A failed run
    keeps the program's environment and whatever artifacts exist, never its metrics.
    """
    try:
        data = _parse_result_document(path)
        status, kind, error = _parse_envelope(data)
    except _ResultError as exc:
        return ExecutionResult(returncode, error=str(exc), failure_kind=UNKNOWN_KIND)
    raw_environment = data.get("environment", {})
    environment = dict(raw_environment) if isinstance(raw_environment, Mapping) else {}
    if returncode == 0 and status == STATUS_OK:
        try:
            metrics = _parse_metrics(data.get("metrics", {}))
            artifacts = _parse_artifacts(data.get("artifacts", []) or [], root)
        except _ResultError as exc:
            return ExecutionResult(
                returncode, environment=environment, error=str(exc), failure_kind=UNKNOWN_KIND
            )
        return ExecutionResult(returncode, metrics, artifacts, environment)
    return ExecutionResult(
        returncode,
        artifacts=_partial_artifacts(data.get("artifacts"), root),
        environment=environment,
        error=error or "the program reported a failure",
        failure_kind=kind or UNKNOWN_KIND,
    )


EXECUTORS.register("subprocess", SubprocessExecutor.from_config)


def build_executor(config: Config, name: str = "subprocess") -> Executor:
    return EXECUTORS.get(name)(config)
