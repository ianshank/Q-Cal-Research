"""RegistryStore: exclusive atomic writes, the hard-link fallback, strict and lenient reads."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import pytest

from qcal.registry import store as store_module
from qcal.registry.records import RecordError
from qcal.registry.store import RecordExistsError, RegistryStore
from tests.conftest import make_record


@pytest.fixture
def store(tmp_path: Path) -> RegistryStore:
    return RegistryStore(tmp_path / "runs" / "registry")


@pytest.fixture
def no_hard_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a filesystem without hard-link support."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("hard links are not supported here")

    monkeypatch.setattr(store_module.os, "link", refuse)


def temp_files(directory: Path) -> list[Path]:
    return sorted(directory.glob(".tmp-*"))


def put(store: RegistryStore, name: str, payload: object) -> Path:
    store.directory.mkdir(parents=True, exist_ok=True)
    path = store.directory / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), "utf-8")
    return path


# -- paths --------------------------------------------------------------------------------


def test_path_for_names_the_file_after_the_run_id(store: RegistryStore) -> None:
    assert store.path_for("R1") == store.directory / "R1.json"


@pytest.mark.parametrize("run_id", ["../escape", "a/b", ".hidden", ""])
def test_path_for_rejects_ids_that_could_escape_the_directory(
    store: RegistryStore, run_id: str
) -> None:
    with pytest.raises(RecordError, match="must match"):
        store.path_for(run_id)


def test_exists_reflects_written_records(store: RegistryStore) -> None:
    before = store.exists("R1")
    store.write(make_record("R1"))
    assert (before, store.exists("R1")) == (False, True)


# -- writing ------------------------------------------------------------------------------


def test_write_creates_the_directory_and_returns_the_path(store: RegistryStore) -> None:
    path = store.write(make_record("R1"))
    assert path == store.directory / "R1.json"
    assert path.is_file()


def test_write_emits_canonical_sorted_json(store: RegistryStore) -> None:
    record = make_record("R1")
    path = store.write(record)
    expected = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    assert path.read_text("utf-8") == expected


def test_write_then_read_round_trips(store: RegistryStore) -> None:
    record = make_record("R1", provenance={"git_sha": "abc"}, environment={"python": "3.11"})
    store.write(record)
    assert store.read("R1") == record


def test_write_refuses_to_replace_an_existing_record(store: RegistryStore) -> None:
    store.write(make_record("R1"))
    with pytest.raises(RecordExistsError, match="R1 already exists"):
        store.write(make_record("R1", status="failed"))


def test_refused_write_leaves_original_record_intact(store: RegistryStore) -> None:
    path = store.write(make_record("R1"))
    original = path.read_text("utf-8")
    with pytest.raises(RecordExistsError):
        store.write(make_record("R1", status="failed"))
    assert path.read_text("utf-8") == original


def test_record_exists_error_is_a_file_exists_error() -> None:
    assert issubclass(RecordExistsError, FileExistsError)


def test_write_leaves_no_temporary_files(store: RegistryStore) -> None:
    store.write(make_record("R1"))
    with pytest.raises(RecordExistsError):
        store.write(make_record("R1"))
    assert temp_files(store.directory) == []


def test_write_stringifies_values_json_cannot_encode(store: RegistryStore) -> None:
    store.write(make_record("R1", environment={"where": Path("/opt/x")}))
    assert store.read("R1").environment["where"] == "/opt/x"


def test_write_falls_back_to_exclusive_create_without_hard_links(
    store: RegistryStore, no_hard_links: None
) -> None:
    record = make_record("R1")
    path = store.write(record)
    assert store.read("R1") == record
    assert path.read_text("utf-8") == json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"


def test_fallback_write_leaves_no_temporary_files(
    store: RegistryStore, no_hard_links: None
) -> None:
    store.write(make_record("R1"))
    assert temp_files(store.directory) == []


def test_fallback_never_overwrites_an_existing_record(
    store: RegistryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.write(make_record("R1"))
    original = path.read_text("utf-8")
    monkeypatch.setattr(store_module.os, "link", _raise_oserror)
    with pytest.raises(RecordExistsError):
        store.write(make_record("R1", status="failed"))
    assert path.read_text("utf-8") == original


def test_fallback_reports_duplicates_as_record_exists_error(
    store: RegistryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write(make_record("R1"))
    monkeypatch.setattr(store_module.os, "link", _raise_oserror)
    with pytest.raises(RecordExistsError, match="R1 already exists"):
        store.write(make_record("R1", status="failed"))


def test_fallback_duplicate_leaves_no_temporary_files(
    store: RegistryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write(make_record("R1"))
    monkeypatch.setattr(store_module.os, "link", _raise_oserror)
    with pytest.raises(RecordExistsError):
        store.write(make_record("R1"))
    assert temp_files(store.directory) == []


def _raise_oserror(*_args: object, **_kwargs: object) -> None:
    raise OSError("operation not supported")


def test_write_uses_hard_link_when_available(
    store: RegistryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    real_link = os.link

    def spy(src: str | Path, dst: str | Path) -> None:
        calls.append((Path(src).name, Path(dst).name))
        real_link(src, dst)

    monkeypatch.setattr(store_module.os, "link", spy)
    store.write(make_record("R1"))
    assert len(calls) == 1
    assert calls[0][0].startswith(".tmp-")
    assert calls[0][1] == "R1.json"


# -- reading ------------------------------------------------------------------------------


def test_iter_records_on_missing_directory_yields_nothing(store: RegistryStore) -> None:
    assert list(store.iter_records(strict=True)) == []


def test_iter_records_yields_records_sorted_by_file_name(store: RegistryStore) -> None:
    for run_id in ("R3", "R1", "R2"):
        store.write(make_record(run_id))
    assert [r.run_id for r in store.iter_records()] == ["R1", "R2", "R3"]


def test_load_all_matches_iter_records(store: RegistryStore) -> None:
    store.write(make_record("R1"))
    store.write(make_record("R2"))
    assert store.load_all() == list(store.iter_records())


def test_iter_records_ignores_hidden_and_temporary_files(store: RegistryStore) -> None:
    store.write(make_record("R1"))
    put(store, ".tmp-abc123.json", "{ partial")
    put(store, ".R2.json", make_record("R2").to_dict())
    assert [r.run_id for r in store.load_all(strict=True)] == ["R1"]


def test_iter_records_ignores_non_json_files(store: RegistryStore) -> None:
    store.write(make_record("R1"))
    put(store, "README.md", "# notes")
    put(store, "R2.json.bak", "not json")
    assert [r.run_id for r in store.load_all(strict=True)] == ["R1"]


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("R2.json", "{ not json"),
        ("R2.json", ["a", "list"]),
        ("R2.json", {"run_id": "R2"}),
        ("R2.json", {**make_record("R2").to_dict(), "seed": "zero"}),
        ("R2.json", {**make_record("R2").to_dict(), "metrics": {"AP": "NaN"}}),
    ],
    ids=["invalid-json", "json-array", "missing-fields", "string-seed", "bad-metric"],
)
def test_lenient_iteration_skips_malformed_records(
    store: RegistryStore, name: str, payload: object
) -> None:
    store.write(make_record("R1"))
    put(store, name, payload)
    assert [r.run_id for r in store.load_all()] == ["R1"]


@pytest.mark.parametrize(
    "payload",
    ["{ not json", ["a", "list"], {"run_id": "R2"}],
    ids=["invalid-json", "json-array", "missing-fields"],
)
def test_strict_iteration_raises_record_error_naming_the_file(
    store: RegistryStore, payload: object
) -> None:
    path = put(store, "R2.json", payload)
    with pytest.raises(RecordError, match=re.escape(str(path))):
        store.load_all(strict=True)


def test_lenient_iteration_logs_skipped_records(
    store: RegistryStore, caplog: pytest.LogCaptureFixture
) -> None:
    put(store, "R2.json", "{ not json")
    with caplog.at_level(logging.WARNING, logger="qcal"):
        store.load_all()
    assert "skipping malformed record" in caplog.text


def test_lenient_iteration_skips_record_whose_id_differs_from_file_name(
    store: RegistryStore, caplog: pytest.LogCaptureFixture
) -> None:
    put(store, "R9.json", make_record("R1").to_dict())
    with caplog.at_level(logging.WARNING, logger="qcal"):
        records = store.load_all()
    assert records == []
    assert "does not match the file name" in caplog.text


def test_strict_iteration_rejects_record_whose_id_differs_from_file_name(
    store: RegistryStore,
) -> None:
    put(store, "R9.json", make_record("R1").to_dict())
    with pytest.raises(RecordError, match="does not match the file name"):
        store.load_all(strict=True)


def test_unreadable_entry_is_skipped_leniently(store: RegistryStore) -> None:
    store.write(make_record("R1"))
    (store.directory / "R2.json").mkdir()
    assert [r.run_id for r in store.load_all()] == ["R1"]


def test_unreadable_entry_fails_strict_iteration(store: RegistryStore) -> None:
    (store.directory / "R2.json").mkdir(parents=True)
    with pytest.raises(RecordError, match=r"R2\.json"):
        store.load_all(strict=True)


def test_read_of_unknown_run_raises_file_not_found(store: RegistryStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.read("R404")


@pytest.mark.parametrize(
    "overrides",
    [{"supersedes": 7}, {"schema_version": "v1"}, {"artifacts": ["x"]}],
    ids=["int-supersedes", "str-schema-version", "non-object-artifact"],
)
def test_lenient_iteration_skips_record_with_mistyped_field(
    store: RegistryStore, overrides: dict[str, object]
) -> None:
    store.write(make_record("R1"))
    put(store, "R2.json", {**make_record("R2").to_dict(), **overrides})
    assert [r.run_id for r in store.load_all()] == ["R1"]


def test_strict_iteration_reports_mistyped_field(store: RegistryStore) -> None:
    put(store, "R2.json", {**make_record("R2").to_dict(), "supersedes": 7})
    with pytest.raises(RecordError, match="supersedes must be a run id string"):
        store.load_all(strict=True)
