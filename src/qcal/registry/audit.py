"""Coverage audit: pre-registered cells x seeds versus what the registry holds.

Replaces the v1 ``ablation-auditor`` agent. It never proposes dropping cells.
"""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.config import Config
from qcal.integrity.aggregates import identity_columns
from qcal.registry.experiments import Experiments
from qcal.registry.index import render_value
from qcal.registry.records import (
    CONFIG_INPUTS_COLUMN,
    EXPERIMENTS_COLUMN,
    POLICY_COLUMN,
    RunRecord,
    effective,
)
from qcal.registry.store import MARKER_SUFFIX, host_identity


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
    duplicates: list[str] = field(default_factory=list)
    bad_supersedes: list[str] = field(default_factory=list)
    mixed_inputs: list[str] = field(default_factory=list)
    uncommitted_policy: list[str] = field(default_factory=list)
    stale_inflight: list[str] = field(default_factory=list)  # report only
    foreign_inflight: list[str] = field(default_factory=list)  # report only: another host
    amendments: int = 0

    def ok(self, *, strict: bool = False) -> bool:
        problems = bool(
            self.unregistered_cells
            or self.unregistered_seeds
            or self.duplicates
            or self.bad_supersedes
        )
        if strict:
            problems = problems or bool(
                self.missing
                or self.failed_only
                or self.placeholders
                or self.mixed_inputs
                or self.uncommitted_policy
            )
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
            "duplicates": self.duplicates,
            "bad_supersedes": self.bad_supersedes,
            "mixed_inputs": self.mixed_inputs,
            "uncommitted_policy": self.uncommitted_policy,
            "stale_inflight": self.stale_inflight,
            "foreign_inflight": self.foreign_inflight,
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
            "duplicates",
            "bad_supersedes",
            "mixed_inputs",
            "uncommitted_policy",
            "stale_inflight",
            "foreign_inflight",
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


def duplicate_pairs(current: Sequence[RunRecord], ok_status: str) -> list[str]:
    """(cell, seed) pairs with more than one current ok run; tables would average them."""
    by_pair: dict[tuple[str, int], list[str]] = {}
    for record in current:
        if record.status == ok_status:
            by_pair.setdefault((record.cell_id, record.seed), []).append(record.run_id)
    return [
        f"{cell}@{seed}: {', '.join(sorted(ids))}"
        for (cell, seed), ids in sorted(by_pair.items())
        if len(ids) > 1
    ]


def bad_supersedes(records: Sequence[RunRecord]) -> list[str]:
    """Supersede links to unknown runs or to a different cell or seed (a typo hides a result)."""
    by_id = {r.run_id: r for r in records}
    problems: list[str] = []
    for record in records:
        if not record.supersedes:
            continue
        cycle = _supersede_cycle(record, by_id)
        if cycle:
            problems.append(f"supersede cycle {' -> '.join(cycle)}")
            continue
        old = by_id.get(record.supersedes)
        if old is None:
            problems.append(f"{record.run_id} supersedes unknown {record.supersedes}")
        elif (old.cell_id, old.seed) != (record.cell_id, record.seed):
            problems.append(
                f"{record.run_id} ({record.cell_id}@{record.seed}) supersedes "
                f"{old.run_id} ({old.cell_id}@{old.seed})"
            )
    return problems


def mixed_inputs(
    current: Sequence[RunRecord],
    ok_status: str,
    *,
    identity: Sequence[str] = (CONFIG_INPUTS_COLUMN,),
    prefixes: Mapping[str, str] | None = None,
) -> list[str]:
    """Cells whose current ok runs differ in an identity column, the policy or the
    pre-registration.

    Tables refuse to aggregate seeds that differ in ``registry.identity_columns`` (the same
    comparison, :func:`qcal.integrity.aggregates.identity_problem`). The audit also reports a
    changed policy or pre-registration, which tables accept: they do not change a number.
    Records without ``provenance.config_inputs_sha256`` (written before it existed) are not
    compared.
    """
    columns = [*identity, POLICY_COLUMN, EXPERIMENTS_COLUMN]
    by_cell: dict[str, dict[str, list[str]]] = {}
    for record in current:
        digest = record.provenance.get(CONFIG_INPUTS_COLUMN)
        if record.status == ok_status and isinstance(digest, str):
            row = record.flat(prefixes or {})
            key = "|".join(render_value(row.get(c)) for c in columns)
            by_cell.setdefault(record.cell_id, {}).setdefault(key, []).append(record.run_id)
    return [
        f"{cell}: {len(digests)} configurations ("
        + "; ".join(", ".join(sorted(ids)) for _, ids in sorted(digests.items()))
        + ")"
        for cell, digests in sorted(by_cell.items())
        if len(digests) > 1
    ]


def uncommitted_policy(current: Sequence[RunRecord], ok_status: str) -> list[str]:
    """Current ok runs whose policy was not verified against a commit (``policy_source``)."""
    return sorted(
        f"{r.run_id}: policy_source={r.provenance['policy_source']}"
        for r in current
        if r.status == ok_status and r.provenance.get("policy_source", "head") != "head"
    )


def inflight_markers(config: Config, records: Sequence[RunRecord]) -> tuple[list[str], list[str]]:
    """(stale, foreign) in-flight markers whose run wrote no record. Report only.

    A launcher holds its marker's lock until the record is written, so a marker nobody locks
    is stale: the launcher died mid-run (power loss, SIGKILL). The audit probes the marker's
    own lock, never the (cell, seed) lock a launcher needs, so it cannot refuse a launcher.
    A marker written on another host or boot cannot be judged by a local lock: foreign.
    """
    directory = config.path("inflight_dir")
    if not directory.is_dir():
        return [], []
    recorded = {r.run_id for r in records}
    here = host_identity()
    stale: list[str] = []
    foreign: list[str] = []
    for marker in sorted(directory.glob(f"*{MARKER_SUFFIX}")):
        run_id = marker.name.removesuffix(MARKER_SUFFIX)
        if run_id in recorded:
            continue
        live = _locked(marker)
        if live is None or live:  # gone (recorded meanwhile) or held by a running launcher
            continue
        try:
            data = json.loads(marker.read_text("utf-8"))
            origin = {k: str(data.get(k, here[k])) for k in here}
        except (OSError, ValueError, AttributeError):
            origin = here  # unreadable: judged by the lock alone
        if origin != here:
            foreign.append(f"{run_id} (host {origin['host']}, boot {origin['boot_id'][:8]})")
        else:
            stale.append(run_id)
    return stale, foreign


def stale_inflight(config: Config, records: Sequence[RunRecord]) -> list[str]:
    """The stale half of :func:`inflight_markers`."""
    return inflight_markers(config, records)[0]


def _locked(marker: Path) -> bool | None:
    """Whether a launcher holds the marker's lock; ``None`` once the marker is gone."""
    try:
        handle = marker.open("rb")
    except FileNotFoundError:
        return None
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def _supersede_cycle(record: RunRecord, by_id: Mapping[str, RunRecord]) -> list[str]:
    """The run ids of a cycle that starts at ``record``, or ``[]``."""
    chain = [record.run_id]
    current: RunRecord | None = record
    while current is not None and current.supersedes:
        if current.supersedes in chain:
            return [*chain, current.supersedes] if current.supersedes == record.run_id else []
        chain.append(current.supersedes)
        current = by_id.get(current.supersedes)
    return []


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
    report.duplicates = duplicate_pairs(current, ok_status)
    report.bad_supersedes = bad_supersedes(records)
    report.mixed_inputs = mixed_inputs(
        current,
        ok_status,
        identity=identity_columns(config),
        prefixes={k: str(v) for k, v in config.section("registry.column_prefixes").items()},
    )
    report.uncommitted_policy = uncommitted_policy(current, ok_status)
    report.stale_inflight, report.foreign_inflight = inflight_markers(config, records)
    for record in current:
        registered = known.get(record.cell_id)
        if registered is None:
            report.unregistered_cells.append(record.run_id)
        elif record.seed not in experiments.seeds_for(registered):
            report.unregistered_seeds.append(record.run_id)
    return report
