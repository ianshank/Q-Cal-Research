"""Coverage audit: pre-registered cells x seeds versus the records in the registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qcal.config import Config
from qcal.registry.audit import AuditReport, audit, count_amendments
from qcal.registry.experiments import Experiments, parse_experiments
from qcal.registry.records import RunRecord
from tests.conftest import make_record, write

DATA: dict[str, Any] = {
    "seed_role": "calibrator_fit_draw",
    "seeds": [0, 1],
    "cells": [{"id": "C-a", "detector": "atss"}, {"id": "C-b", "detector": "detr", "seeds": [5]}],
}
ALL_PAIRS = [("C-a", 0), ("C-a", 1), ("C-b", 5)]


def experiments_from(config: Config, **overrides: Any) -> Experiments:
    return parse_experiments(config, {**DATA, **overrides}, path=config.path("experiments"))


@pytest.fixture
def experiments(config: Config) -> Experiments:
    return experiments_from(config)


def ok(run_id: str, cell: str, seed: int, **fields: Any) -> RunRecord:
    return make_record(run_id, cell_id=cell, seed=seed, **fields)


def failed(run_id: str, cell: str, seed: int, **fields: Any) -> RunRecord:
    return make_record(run_id, cell_id=cell, seed=seed, status="failed", metrics={}, **fields)


def complete() -> list[RunRecord]:
    return [ok(f"R{i}", cell, seed) for i, (cell, seed) in enumerate(ALL_PAIRS)]


# -- coverage -----------------------------------------------------------------------------


def test_full_coverage(config: Config, experiments: Experiments) -> None:
    report = audit(config, experiments, complete())
    assert (report.expected, report.completed, report.missing, report.failed_only) == (
        3,
        3,
        [],
        [],
    )
    assert report.ok(strict=True)


def test_expected_counts_cell_specific_seeds(config: Config, experiments: Experiments) -> None:
    assert audit(config, experiments, []).expected == 3


def test_empty_registry_reports_every_pair_missing(
    config: Config, experiments: Experiments
) -> None:
    report = audit(config, experiments, [])
    assert (report.completed, report.missing) == (0, ALL_PAIRS)


def test_pair_with_only_failed_runs_is_failed_only(
    config: Config, experiments: Experiments
) -> None:
    report = audit(config, experiments, [*complete()[1:], failed("F1", "C-a", 0)])
    assert (report.completed, report.failed_only, report.missing) == (2, [("C-a", 0)], [])


def test_any_ok_run_completes_a_pair(config: Config, experiments: Experiments) -> None:
    records = [failed("F1", "C-a", 0), ok("R1", "C-a", 0)]
    report = audit(config, experiments, records)
    assert report.completed == 1
    assert report.failed_only == []


def test_duplicate_ok_runs_count_once(config: Config, experiments: Experiments) -> None:
    records = [ok("R1", "C-a", 0), ok("R2", "C-a", 0)]
    assert audit(config, experiments, records).completed == 1


def test_superseded_ok_run_does_not_count(config: Config, experiments: Experiments) -> None:
    records = [ok("R1", "C-a", 0), failed("R2", "C-a", 0, supersedes="R1")]
    report = audit(config, experiments, records)
    assert (report.completed, report.failed_only) == (0, [("C-a", 0)])


def test_superseding_ok_run_counts(config: Config, experiments: Experiments) -> None:
    records = [ok("R1", "C-a", 0), ok("R2", "C-a", 0, supersedes="R1")]
    assert audit(config, experiments, records).completed == 1


def test_superseded_runs_are_listed_sorted_and_unique(
    config: Config, experiments: Experiments
) -> None:
    records = [
        ok("R3", "C-a", 0),
        ok("R5", "C-a", 0, supersedes="R3"),
        ok("R4", "C-a", 1, supersedes="R1"),
        ok("R6", "C-a", 1, supersedes="R1"),
    ]
    assert audit(config, experiments, records).superseded == ["R1", "R3"]


def test_configured_ok_status_defines_completion(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config('[registry]\nok_status = "done"\n')
    records = [ok("R1", "C-a", 0, status="done"), ok("R2", "C-a", 1, status="ok")]
    report = audit(config, experiments_from(config), records)
    assert (report.completed, report.failed_only) == (1, [("C-a", 1)])


# -- unregistered records -----------------------------------------------------------------


def test_records_for_unregistered_cells_are_flagged(
    config: Config, experiments: Experiments
) -> None:
    report = audit(config, experiments, [*complete(), ok("X1", "C-zzz", 0)])
    assert report.unregistered_cells == ["X1"]
    assert report.ok() is False


def test_records_for_unregistered_seeds_are_flagged(
    config: Config, experiments: Experiments
) -> None:
    records = [*complete(), ok("X1", "C-a", 5), ok("X2", "C-b", 0)]
    report = audit(config, experiments, records)
    assert report.unregistered_seeds == ["X1", "X2"]
    assert report.ok() is False


def test_unregistered_failed_runs_are_flagged_too(
    config: Config, experiments: Experiments
) -> None:
    report = audit(config, experiments, [failed("X1", "C-zzz", 0)])
    assert report.unregistered_cells == ["X1"]


def test_superseded_unregistered_records_are_not_flagged(
    config: Config, experiments: Experiments
) -> None:
    records = [ok("X1", "C-zzz", 0), ok("R1", "C-a", 0, supersedes="X1")]
    assert audit(config, experiments, records).unregistered_cells == []


# -- placeholders and amendments ----------------------------------------------------------


def test_placeholders_come_from_the_pre_registration(config: Config) -> None:
    experiments = experiments_from(config, question={"ian": "x"}, seed_role="ian: role")
    assert audit(config, experiments, complete()).placeholders == ["seed_role", "question.ian"]


def test_amendments_are_counted_from_dated_headings(config: Config, repo: Path) -> None:
    write(
        repo,
        "AMENDMENTS.md",
        """\
        # Amendments

        ## 2026-10-01 drop the fog corruption
        text
        ## 2026-10-05: add a seed
        ### 2026-10-06 a sub-heading is not an amendment
        ## Undated heading
          ## 2026-10-07 indented headings do not count
        """,
    )
    assert count_amendments(config) == 2


def test_missing_amendments_file_counts_zero(config: Config) -> None:
    assert count_amendments(config) == 0


def test_amendment_pattern_and_path_are_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config(
        """
        [paths]
        amendments = "docs/AMEND.md"

        [experiments]
        amendment_heading_pattern = '^- amendment'
        """
    )
    write(repo, "docs/AMEND.md", "- amendment one\n- amendment two\n- note\n")
    assert count_amendments(config) == 2


def test_audit_reports_amendment_count(
    config: Config, experiments: Experiments, repo: Path
) -> None:
    write(repo, "AMENDMENTS.md", "## 2026-10-01 one\n")
    assert audit(config, experiments, []).amendments == 1


# -- verdicts -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "lenient", "strict"),
    [
        ({}, True, True),
        ({"missing": [("C-a", 0)]}, True, False),
        ({"placeholders": ["seed_role"]}, True, False),
        ({"failed_only": [("C-a", 0)]}, True, True),
        ({"superseded": ["R1"]}, True, True),
        ({"amendments": 3}, True, True),
        ({"unregistered_cells": ["X1"]}, False, False),
        ({"unregistered_seeds": ["X1"]}, False, False),
    ],
)
def test_ok_verdicts(fields: dict[str, Any], lenient: bool, strict: bool) -> None:
    report = AuditReport(**fields)
    assert (report.ok(), report.ok(strict=True)) == (lenient, strict)


def test_empty_registry_passes_lenient_but_fails_strict(
    config: Config, experiments: Experiments
) -> None:
    report = audit(config, experiments, [])
    assert (report.ok(), report.ok(strict=True)) == (True, False)


# -- rendering ----------------------------------------------------------------------------


def test_to_dict_formats_pairs_and_coverage(config: Config, experiments: Experiments) -> None:
    records = [ok("R1", "C-a", 0), failed("F1", "C-a", 1)]
    assert audit(config, experiments, records).to_dict() == {
        "expected": 3,
        "completed": 1,
        "coverage": 0.3333,
        "missing": ["C-b@5"],
        "failed_only": ["C-a@1"],
        "unregistered_cells": [],
        "unregistered_seeds": [],
        "superseded": [],
        "placeholders": [],
        "amendments": 0,
    }


def test_coverage_is_none_without_expected_pairs() -> None:
    assert AuditReport().to_dict()["coverage"] is None


def test_render_text_lists_every_section() -> None:
    report = AuditReport(
        expected=3,
        completed=1,
        missing=[("C-b", 5)],
        failed_only=[("C-a", 1)],
        superseded=["R0"],
        placeholders=["seed_role"],
        amendments=2,
    )
    assert report.render_text().splitlines() == [
        "coverage: 1/3 cell-seed pairs",
        "missing: 1",
        "  - C-b@5",
        "failed_only: 1",
        "  - C-a@1",
        "unregistered_cells: 0",
        "unregistered_seeds: 0",
        "superseded: 1",
        "  - R0",
        "placeholders: 1",
        "  - seed_role",
        "amendments: 2",
    ]
