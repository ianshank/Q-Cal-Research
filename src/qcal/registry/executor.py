"""How a registered run is actually executed.

The default :class:`SubprocessExecutor` runs an argv template from configuration
and reads a result JSON the experiment program writes:

    {"metrics": {"AP": 41.2, ...},
     "artifacts": [{"path": "runs/results/x.engine", "kind": "engine"}],
     "environment": {"trt_version": "10.3.0"}}

Templates may use only the documented placeholders, and the CLI accepts no
free-form overrides, so a run cannot drift from the pre-registered cell. ``{python}`` is
the launcher's own interpreter, so the hashed template names no machine path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import string
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from qcal.components import ComponentRegistry
from qcal.config import Config, ConfigError, run_neutral_environment, without_config_environment
from qcal.log import get_logger
from qcal.registry.experiments import Cell
from qcal.registry.records import METRIC_NAME_PATTERN, ArtifactRef

_log = get_logger("registry.executor")

PLACEHOLDERS: Final = frozenset(
    {"python", "run_id", "cell_id", "seed", "seed_role", "result_path", "log_path", "root"}
)
_HASH_CHUNK: Final = 1 << 20


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

    @classmethod
    def from_config(cls, config: Config) -> SubprocessExecutor:
        timeout = config.float_value("executor.timeout_s")
        return cls(
            config.str_list("executor.command"),
            timeout_s=timeout if timeout > 0 else None,
            env_prefix=config.str_value("executor.env_prefix"),
            root=config.root,
            pass_env=run_neutral_environment(config),
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
        try:
            with spec.log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    argv,
                    cwd=self.root,
                    env=self.environment(spec),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(returncode=-1, error=f"timed out after {self.timeout_s}s")
        except OSError as exc:
            return ExecutionResult(returncode=-1, error=f"cannot start {argv[0]}: {exc}")
        if proc.returncode != 0:
            return ExecutionResult(
                proc.returncode, error=f"exit code {proc.returncode}; see {spec.log_path}"
            )
        return read_result(spec.result_path, self.root, proc.returncode)


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


def read_result(path: Path, root: Path, returncode: int = 0) -> ExecutionResult:
    """Parse and validate the result JSON an experiment wrote."""
    try:
        data = _parse_result_document(path)
        metrics = _parse_metrics(data.get("metrics", {}))
        artifacts = _parse_artifacts(data.get("artifacts", []) or [], root)
    except _ResultError as exc:
        return ExecutionResult(returncode, error=str(exc))
    environment = data.get("environment", {})
    return ExecutionResult(
        returncode,
        metrics=metrics,
        artifacts=artifacts,
        environment=dict(environment) if isinstance(environment, Mapping) else {},
    )


EXECUTORS.register("subprocess", SubprocessExecutor.from_config)


def build_executor(config: Config, name: str = "subprocess") -> Executor:
    return EXECUTORS.get(name)(config)
