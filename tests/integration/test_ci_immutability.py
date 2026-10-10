"""Registry immutability CI check: between base and head, records may only be added, and
every added record must be valid and named after its run id."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qcal.ci import immutability
from qcal.ci.immutability import ImmutabilityReport, check_registry_immutable
from qcal.registry.records import RunRecord
from qcal.registry.store import RegistryStore
from tests.conftest import make_record, run_git, write

pytestmark = pytest.mark.integration

REGISTRY = "runs/registry"
R1 = f"{REGISTRY}/R1.json"
R2 = f"{REGISTRY}/R2.json"


def commit(root: Path, message: str) -> str:
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "--allow-empty", "-m", message)
    return run_git(root, "rev-parse", "HEAD")


def store(root: Path) -> RegistryStore:
    return RegistryStore(root / REGISTRY)


def record_json(record: RunRecord, **overrides: Any) -> str:
    data = record.to_dict()
    data.update(overrides)
    return json.dumps(data, allow_nan=True)


def check(root: Path, **kwargs: Any) -> ImmutabilityReport:
    return check_registry_immutable(root, "main", "feature", **kwargs)


@pytest.fixture
def make_base(git_repo: Path) -> Callable[[str | None], Path]:
    """Commit record R1 (and optionally a qcal.toml policy) on main, then branch ``feature``."""

    def _make(policy: str | None = None) -> Path:
        if policy is not None:
            write(git_repo, "qcal.toml", policy)
        store(git_repo).write(make_record("R1"))
        commit(git_repo, "base record")
        run_git(git_repo, "checkout", "-q", "-b", "feature")
        return git_repo

    return _make


@pytest.fixture
def project(make_base: Callable[[str | None], Path]) -> Path:
    return make_base(None)


# --- additions ----------------------------------------------------------------------------


def test_adding_a_valid_record_passes(project: Path) -> None:
    store(project).write(make_record("R2"))
    commit(project, "add R2")

    report = check(project)

    assert report.passed
    assert (report.added, report.violations) == ([R2], [])


def test_several_additions_are_listed_in_path_order(project: Path) -> None:
    for run_id in ("R3", "R2"):
        store(project).write(make_record(run_id))
        commit(project, f"add {run_id}")

    assert check(project).added == [R2, f"{REGISTRY}/R3.json"]


def test_no_changes_passes(project: Path) -> None:
    report = check(project)

    assert report.passed
    assert report.added == []


def test_added_non_json_file_is_a_violation(project: Path) -> None:
    write(project, f"{REGISTRY}/README.txt", "not a record\n")
    commit(project, "notes")

    report = check(project)

    assert not report.passed
    assert report.violations == [
        f"{REGISTRY}/README.txt: only <run_id>.json records belong in the registry directory"
    ]


def test_changes_outside_immutable_dirs_are_ignored(project: Path) -> None:
    write(project, "runs/index.csv", "run_id\n")
    write(project, "runs/registry_backup/R1.json", "garbage")
    write(project, "notes.md", "x\n")
    commit(project, "elsewhere")

    report = check(project)

    assert report.passed
    assert report.added == []


# --- existing records are append-only ---------------------------------------------------------


def _modify(root: Path) -> None:
    write(root, R1, record_json(make_record("R1"), metrics={"AP": 99.0}))


def _delete(root: Path) -> None:
    (root / R1).unlink()


def _chmod(root: Path) -> None:
    (root / R1).chmod(0o755)


def _symlink(root: Path) -> None:
    (root / R1).unlink()
    (root / R1).symlink_to("R1-elsewhere.json")


@pytest.mark.parametrize(
    ("mutate", "status"),
    [(_modify, "M"), (_delete, "D"), (_chmod, "M"), (_symlink, "T")],
    ids=["modify", "delete", "mode-change", "type-change"],
)
def test_existing_record_may_not_change(
    project: Path, mutate: Callable[[Path], None], status: str
) -> None:
    mutate(project)
    commit(project, "tamper")

    report = check(project)

    assert not report.passed
    assert report.violations == [f"{R1}: status {status} (records are append-only)"]


def test_renaming_a_record_is_a_deletion_plus_an_invalid_addition(project: Path) -> None:
    run_git(project, "mv", R1, f"{REGISTRY}/R1b.json")
    commit(project, "rename")

    report = check(project)

    assert report.violations == [
        f"{R1}: status D (records are append-only)",
        f"{REGISTRY}/R1b.json: run_id 'R1' does not match the file name",
    ]


def test_record_modified_and_restored_within_the_branch_passes(project: Path) -> None:
    original = (project / R1).read_text("utf-8")
    _modify(project)
    commit(project, "tamper")
    write(project, R1, original)
    commit(project, "restore")

    assert check(project).passed


# --- validation of added JSON -----------------------------------------------------------------


_R2 = make_record("R2")


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        pytest.param("not json", "Expecting value", id="not-json"),
        pytest.param("[]", "record is missing run_id", id="list"),
        pytest.param("5", "malformed record", id="number"),
        pytest.param("null", "malformed record", id="null"),
        pytest.param(
            '{"run_id": "R2"}',
            "record is missing cell_id, seed, status, started_at, finished_at",
            id="missing-fields",
        ),
        pytest.param(record_json(_R2, seed="0"), "seed must be an integer", id="seed-type"),
        pytest.param(record_json(_R2, metrics=[]), "metrics must be an object", id="metrics-type"),
        pytest.param(
            record_json(_R2, schema_version="one"),
            "schema_version must be an integer",
            id="schema-version",
        ),
        pytest.param(
            record_json(_R2, artifacts=["x"]), "artifacts must be a list of objects", id="artifacts"
        ),
        pytest.param(
            record_json(_R2, metrics={"AP": float("nan")}), "must be a finite number", id="nan"
        ),
        pytest.param(
            record_json(_R2, metrics={"bad name": 1.0}), "metric name 'bad name'", id="metric-name"
        ),
        pytest.param(record_json(_R2, supersedes=5), "supersedes must be", id="supersedes"),
    ],
)
def test_added_invalid_record_is_a_violation(project: Path, payload: str, detail: str) -> None:
    write(project, R2, payload)
    commit(project, "bad record")

    report = check(project)

    assert report.added == [R2]
    assert len(report.violations) == 1
    assert report.violations[0].startswith(f"{R2}: invalid record (")
    assert detail in report.violations[0]


def test_added_record_must_be_named_after_its_run_id(project: Path) -> None:
    write(project, R2, record_json(make_record("R3")))
    commit(project, "misnamed")

    assert check(project).violations == [f"{R2}: run_id 'R3' does not match the file name"]


def test_added_record_is_read_from_the_head_commit(project: Path) -> None:
    store(project).write(make_record("R2"))
    commit(project, "add R2")
    write(project, R2, "uncommitted garbage")

    assert check(project).passed


def test_blank_lines_in_git_output_are_skipped(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store(project).write(make_record("R2"))
    commit(project, "add R2")
    real_git = immutability.git
    monkeypatch.setattr(immutability, "git", lambda *a, **k: "\n" + real_git(*a, **k) + "\n  \n")

    report = check(project)

    assert (report.added, report.violations) == ([R2], [])


def test_unreadable_added_record_is_a_violation(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store(project).write(make_record("R2"))
    commit(project, "add R2")
    monkeypatch.setattr(immutability, "show_file", lambda *_: None)

    assert check(project).violations == [f"{R2}: cannot read added record"]


# --- which commits and which policy -----------------------------------------------------------


def test_changes_on_base_after_the_fork_are_not_attributed_to_head(project: Path) -> None:
    store(project).write(make_record("R2"))
    commit(project, "add R2")
    run_git(project, "checkout", "-q", "main")
    _modify(project)
    store(project).write(make_record("R3"))
    commit(project, "maintenance on main")
    run_git(project, "checkout", "-q", "feature")

    report = check(project)

    assert report.passed
    assert report.added == [R2]


def test_head_relaxing_immutable_dirs_is_judged_by_the_base_policy(project: Path) -> None:
    write(project, "qcal.toml", "[signing]\nimmutable_dirs = []\n")
    _modify(project)
    commit(project, "relax and tamper")

    assert check(project).violations == [f"{R1}: status M (records are append-only)"]


def test_base_policy_without_immutable_dirs_checks_nothing(
    make_base: Callable[[str | None], Path],
) -> None:
    root = make_base("[signing]\nimmutable_dirs = []\n")
    _modify(root)
    commit(root, "tamper")

    report = check(root)

    assert report.passed
    assert (report.added, report.violations) == ([], [])


def test_immutable_dirs_come_from_the_base_policy(
    git_repo: Path, make_base: Callable[[str | None], Path]
) -> None:
    write(git_repo, "records/R9.json", record_json(make_record("R9")))
    root = make_base('[signing]\nimmutable_dirs = ["records"]\n')
    _modify(root)
    write(root, "records/R9.json", record_json(make_record("R9"), status="failed"))
    commit(root, "tamper")

    assert check(root).violations == ["records/R9.json: status M (records are append-only)"]


def test_policy_ref_selects_the_policy(make_base: Callable[[str | None], Path]) -> None:
    root = make_base(None)
    run_git(root, "tag", "strict", "main")
    run_git(root, "checkout", "-q", "main")
    write(root, "qcal.toml", "[signing]\nimmutable_dirs = []\n")
    commit(root, "relax on main")
    run_git(root, "checkout", "-q", "-B", "feature")
    _modify(root)
    commit(root, "tamper")

    assert check(root).passed
    assert check(root, policy_ref="strict").violations == [
        f"{R1}: status M (records are append-only)"
    ]


# --- report -------------------------------------------------------------------------------


def test_report_dict(project: Path) -> None:
    store(project).write(make_record("R2"))
    (project / R1).unlink()
    commit(project, "mixed")

    assert check(project).to_dict() == {
        "verdict": "FAIL",
        "added": [R2],
        "violations": [f"{R1}: status D (records are append-only)"],
    }


def test_empty_report_passes() -> None:
    assert ImmutabilityReport().to_dict() == {"verdict": "PASS", "added": [], "violations": []}
