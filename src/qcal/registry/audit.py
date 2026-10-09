"""Coverage audit: pre-registered cells x seeds versus what the registry holds.

Replaces the v1 ``ablation-auditor`` agent. It never proposes dropping cells.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from qcal.config import Config
from qcal.registry.experiments import Experiments
from qcal.registry.records import RunRecord, effective


@dataclass
class AuditReport:
    expected: int = 0
    completed: int = 0
    missing: list[tuple[str, int]] = field(default_factory=list)
    failed_only: list[tuple[str, int]] = field(default_factory=list)
    unregistered_cells: list[str] = field(default_factory=list)
    unregistered_seeds: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    amendments: int = 0

    def ok(self, *, strict: bool = False) -> bool:
        problems = bool(self.unregistered_cells or self.unregistered_seeds)
        if strict:
            problems = problems or bool(self.missing or self.placeholders)
        return not problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": self.expected,
            "completed": self.completed,
            "coverage": round(self.completed / self.expected, 4) if self.expected else None,
            "missing": [f"{c}@{s}" for c, s in self.missing],
            "failed_only": [f"{c}@{s}" for c, s in self.failed_only],
            "unregistered_cells": self.unregistered_cells,
            "unregistered_seeds": self.unregistered_seeds,
            "superseded": self.superseded,
            "placeholders": self.placeholders,
            "amendments": self.amendments,
        }

    def render_text(self) -> str:
        data = self.to_dict()
        lines = [f"coverage: {self.completed}/{self.expected} cell-seed pairs"]
        for key in (
            "missing",
            "failed_only",
            "unregistered_cells",
            "unregistered_seeds",
            "superseded",
            "placeholders",
        ):
            values = data[key]
            lines.append(f"{key}: {len(values)}")
            lines.extend(f"  - {v}" for v in values)
        lines.append(f"amendments: {self.amendments}")
        return "\n".join(lines)


def count_amendments(config: Config) -> int:
    path = config.path("amendments")
    if not path.is_file():
        return 0
    pattern = re.compile(config.str_value("experiments.amendment_heading_pattern"), re.MULTILINE)
    return len(pattern.findall(path.read_text("utf-8")))


def audit(config: Config, experiments: Experiments, records: Sequence[RunRecord]) -> AuditReport:
    ok_status = config.str_value("registry.ok_status")
    current = effective(list(records))
    superseded = sorted({r.supersedes for r in records if r.supersedes})
    known = {c.id: c for c in experiments.cells}
    report = AuditReport(
        superseded=superseded,
        placeholders=list(experiments.placeholders),
        amendments=count_amendments(config),
    )
    ok_pairs = {(r.cell_id, r.seed) for r in current if r.status == ok_status}
    any_pairs = {(r.cell_id, r.seed) for r in current}
    for cell in experiments.cells:
        for seed in experiments.seeds_for(cell):
            report.expected += 1
            if (cell.id, seed) in ok_pairs:
                report.completed += 1
            elif (cell.id, seed) in any_pairs:
                report.failed_only.append((cell.id, seed))
            else:
                report.missing.append((cell.id, seed))
    for record in current:
        registered = known.get(record.cell_id)
        if registered is None:
            report.unregistered_cells.append(record.run_id)
        elif record.seed not in experiments.seeds_for(registered):
            report.unregistered_seeds.append(record.run_id)
    return report
