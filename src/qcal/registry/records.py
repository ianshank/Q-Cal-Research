"""The run record: one immutable JSON document per completed run."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

RUN_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: Metric names appear inside claim references (``run:<id>:<metric>``) and inside LaTeX
#: macro arguments, so they are limited to letters, digits and ``_ . @ -``.
METRIC_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]*$")
_REQUIRED: Final = ("run_id", "cell_id", "seed", "status", "started_at", "finished_at")
#: The record schema this code writes and the newest it reads. Schema 1 only ever gains
#: optional fields; a change that old readers would misread bumps it, and old readers then
#: refuse the new records instead of misreading them.
SCHEMA_VERSION: Final = 1
_REQUIRED_STRINGS: Final = ("run_id", "cell_id", "status", "started_at", "finished_at")


class RecordError(ValueError):
    """A run record is malformed."""


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RecordError(f"run id {run_id!r} must match {RUN_ID_PATTERN.pattern}")
    return run_id


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    kind: str = ""
    size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "kind": self.kind, "size": self.size}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRef:
        if not isinstance(data.get("path"), str) or not isinstance(data.get("sha256"), str):
            raise RecordError("artifact entries need string 'path' and 'sha256'")
        size = data.get("size")
        return cls(
            data["path"],
            data["sha256"],
            str(data.get("kind", "")),
            size if isinstance(size, int) else None,
        )


#: Why a run failed, as the executor result envelope and a failed record name it.
FAILURE_KINDS: Final = (
    "timeout",
    "cuda_oom",
    "host_oom",
    "cuda_error",
    "plan",
    "config",
    "interrupted",
    "unknown",
)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    cell_id: str
    seed: int
    status: str
    started_at: str
    finished_at: str
    seed_role: str = ""
    supersedes: str | None = None
    duration_s: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    factors: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    log_path: str | None = None
    error: str | None = None
    failure_kind: str | None = None  # one of FAILURE_KINDS for a failed run (schema 1, added)
    schema_version: int = SCHEMA_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.supersedes is not None:
            validate_run_id(self.supersedes)
            if self.supersedes == self.run_id:
                raise RecordError(f"run {self.run_id} cannot supersede itself")
        for name, value in self.metrics.items():
            if not METRIC_NAME_PATTERN.fullmatch(name):
                raise RecordError(f"metric name {name!r} must match {METRIC_NAME_PATTERN.pattern}")
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise RecordError(f"metric {name!r} must be a finite number, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "seed": self.seed,
            "seed_role": self.seed_role,
            "status": self.status,
            "supersedes": self.supersedes,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "provenance": dict(self.provenance),
            "factors": dict(self.factors),
            "metrics": dict(self.metrics),
            "environment": dict(self.environment),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "log_path": self.log_path,
            "error": self.error,
        }
        if self.failure_kind is not None:  # absent from records written before it existed
            data["failure_kind"] = self.failure_kind
        data.update({k: v for k, v in self.extra.items() if k not in data})
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunRecord:
        """Build a record from JSON data; every type problem is a :class:`RecordError`."""
        try:
            return cls._from_dict(data)
        except RecordError:
            raise
        except (TypeError, ValueError, AttributeError) as exc:
            raise RecordError(f"malformed record: {exc}") from exc

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> RunRecord:
        missing = [k for k in _REQUIRED if k not in data]
        if missing:
            raise RecordError(f"record is missing {', '.join(missing)}")
        for key in _REQUIRED_STRINGS:
            value = data[key]
            if not isinstance(value, str) or not value.strip():
                raise RecordError(f"{key} must be a non-empty string, got {value!r}")
        seed_role = data.get("seed_role")
        if seed_role is not None and not isinstance(seed_role, str):
            raise RecordError(f"seed_role must be a string, got {seed_role!r}")
        seed = data["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RecordError("seed must be an integer")
        for key in ("provenance", "factors", "metrics", "environment"):
            if not isinstance(data.get(key, {}), Mapping):
                raise RecordError(f"{key} must be an object")
        artifacts = data.get("artifacts", [])
        if not isinstance(artifacts, list) or not all(isinstance(a, Mapping) for a in artifacts):
            raise RecordError("artifacts must be a list of objects")
        failure_kind = data.get("failure_kind")
        if failure_kind is not None and not isinstance(failure_kind, str):
            raise RecordError("failure_kind must be a string or null")
        supersedes = data.get("supersedes")
        if supersedes is not None and not isinstance(supersedes, str):
            raise RecordError("supersedes must be a run id string or null")
        version = data.get("schema_version", 1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise RecordError("schema_version must be an integer")
        if not 1 <= version <= SCHEMA_VERSION:
            raise RecordError(
                f"schema_version {version} is not readable by this qcal (it reads 1 to "
                f"{SCHEMA_VERSION}); a newer record needs a newer qcal"
            )
        known = set(cls.__dataclass_fields__) - {"extra"}
        duration = data.get("duration_s")
        return cls(
            run_id=data["run_id"],
            cell_id=data["cell_id"],
            seed=seed,
            status=data["status"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            seed_role=seed_role or "",
            supersedes=supersedes or None,
            duration_s=float(duration) if isinstance(duration, int | float) else None,
            provenance=dict(data.get("provenance", {})),
            factors=dict(data.get("factors", {})),
            metrics=dict(data.get("metrics", {})),
            environment=dict(data.get("environment", {})),
            artifacts=tuple(ArtifactRef.from_dict(a) for a in artifacts),
            log_path=data.get("log_path"),
            error=data.get("error"),
            failure_kind=failure_kind,
            schema_version=version,
            extra={k: v for k, v in data.items() if k not in known},
        )

    def flat(self, prefixes: Mapping[str, str]) -> dict[str, Any]:
        """Flatten into one index row; nested groups get configurable column prefixes."""
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "seed": self.seed,
            "seed_role": self.seed_role,
            "status": self.status,
            "supersedes": self.supersedes,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "git_sha": self.provenance.get("git_sha"),
            "git_dirty": self.provenance.get("git_dirty"),
            "config_hash": self.provenance.get("config_hash"),
        }
        for group, values in (
            ("factors", self.factors),
            ("metrics", self.metrics),
            ("environment", self.environment),
        ):
            prefix = prefixes.get(group, f"{group}.")
            for key, value in values.items():
                row[f"{prefix}{key}"] = value
        return row


def effective(records: list[RunRecord]) -> list[RunRecord]:
    """Records not superseded by a later record."""
    superseded = {r.supersedes for r in records if r.supersedes}
    return [r for r in records if r.run_id not in superseded]
