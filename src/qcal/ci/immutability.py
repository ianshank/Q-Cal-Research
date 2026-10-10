"""Registry immutability: between base and head, records may only be added."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.ci.signatures import policy_config, require_commit
from qcal.gitutil import git, show_file
from qcal.log import get_logger
from qcal.registry.records import RecordError, RunRecord
from qcal.reports import verdict

_log = get_logger("ci.immutability")


@dataclass
class ImmutabilityReport:
    added: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": verdict(self.passed),
            "added": self.added,
            "violations": self.violations,
        }

    def render_text(self) -> str:
        head = (
            f"registry immutability: {'PASS' if self.passed else 'FAIL'} ({len(self.added)} added)"
        )
        return "\n".join([head, *(f"  {v}" for v in self.violations)])


def check_registry_immutable(
    repo: Path, base: str, head: str, *, policy_ref: str | None = None
) -> ImmutabilityReport:
    base = require_commit(repo, base, "base")
    head = require_commit(repo, head, "head")
    config = policy_config(repo, policy_ref or base)
    report = ImmutabilityReport()
    dirs = config.str_list("signing.immutable_dirs")
    if not dirs:
        return report
    out = git(
        ["diff", "-z", "--name-status", "--no-renames", f"{base}...{head}", "--", *dirs], repo
    )
    # Records are named by run id ([A-Za-z0-9._-]), so surrounding whitespace is never meaningful.
    fields = [f.strip() for f in out.split("\0") if f.strip()]
    if len(fields) % 2:
        report.violations.append("unparseable git diff output; refusing to judge (fail closed)")
        fields = fields[:-1]
    for status, path in zip(fields[0::2], fields[1::2], strict=True):
        if status != "A":
            report.violations.append(f"{path}: status {status} (records are append-only)")
            continue
        report.added.append(path)
        problem = _validate_added(repo, head, path, dirs)
        if problem:
            report.violations.append(f"{path}: {problem}")
    _log.info(
        "registry immutability: %s (%d added)",
        verdict(report.passed),
        len(report.added),
    )
    return report


def _registry_parent(path: str, dirs: list[str]) -> str | None:
    """The registry directory ``path`` sits directly in, or ``None`` (nested or elsewhere)."""
    for directory in (d.rstrip("/") for d in dirs):
        if path.startswith(directory + "/") and "/" not in path[len(directory) + 1 :]:
            return directory
    return None


def _validate_added(repo: Path, head: str, path: str, dirs: list[str]) -> str | None:
    parent = _registry_parent(path, dirs)
    if parent is None:
        return "records live directly in the registry directory (the store ignores subfolders)"
    if not path.endswith(".json"):
        return "only <run_id>.json records belong in the registry directory"
    return _validate_record(repo, head, path, parent)


def _validate_record(repo: Path, head: str, path: str, parent: str) -> str | None:
    text = show_file(head, path, repo)
    if text is None:
        return "cannot read added record"
    try:
        record = RunRecord.from_dict(json.loads(text))
    except (json.JSONDecodeError, RecordError, TypeError) as exc:
        return f"invalid record ({exc})"
    if record.run_id != Path(path).stem:
        return f"run_id {record.run_id!r} does not match the file name"
    if record.supersedes:
        return _check_supersedes(repo, head, parent, record)
    return None


def _check_supersedes(repo: Path, head: str, parent: str, record: RunRecord) -> str | None:
    text = show_file(head, f"{parent}/{record.supersedes}.json", repo)
    if text is None:
        return f"supersedes {record.supersedes!r}, which is not in the registry"
    try:
        old = RunRecord.from_dict(json.loads(text))
    except (json.JSONDecodeError, RecordError, TypeError) as exc:
        return f"supersedes {record.supersedes!r}, which is not a valid record ({exc})"
    if (old.cell_id, old.seed) != (record.cell_id, record.seed):
        return (
            f"supersedes {record.supersedes!r} of {old.cell_id}@{old.seed}, "
            f"not {record.cell_id}@{record.seed}"
        )
    return None
