"""Registry immutability: between base and head, records may only be added."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.ci.signatures import policy_config
from qcal.gitutil import git, show_file
from qcal.log import get_logger
from qcal.registry.records import RecordError, RunRecord

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
            "verdict": "PASS" if self.passed else "FAIL",
            "added": self.added,
            "violations": self.violations,
        }


def check_registry_immutable(
    repo: Path, base: str, head: str, *, policy_ref: str | None = None
) -> ImmutabilityReport:
    config = policy_config(repo, policy_ref or base)
    report = ImmutabilityReport()
    dirs = config.str_list("signing.immutable_dirs")
    if not dirs:
        return report
    out = git(["diff", "--name-status", "--no-renames", f"{base}...{head}", "--", *dirs], repo)
    for line in out.splitlines():
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        if status != "A":
            report.violations.append(f"{path}: status {status} (records are append-only)")
            continue
        report.added.append(path)
        problem = _validate_added(repo, head, path)
        if problem:
            report.violations.append(f"{path}: {problem}")
    _log.info(
        "registry immutability: %s (%d added)",
        "PASS" if report.passed else "FAIL",
        len(report.added),
    )
    return report


def _validate_added(repo: Path, head: str, path: str) -> str | None:
    if not path.endswith(".json"):
        return None
    text = show_file(head, path, repo)
    if text is None:
        return "cannot read added record"
    try:
        record = RunRecord.from_dict(json.loads(text))
    except (json.JSONDecodeError, RecordError, TypeError) as exc:
        return f"invalid record ({exc})"
    if record.run_id != Path(path).stem:
        return f"run_id {record.run_id!r} does not match the file name"
    return None
