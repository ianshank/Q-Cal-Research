"""Copilot review of PR #2: one test per finding."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from qcal.registry.cells import expand_design
from qcal.registry.experiments import ExperimentsError
from qcal.registry.records import RecordError, RunRecord
from qcal.registry.store import RegistryStore
from tests.conftest import REPO_ROOT, make_record


def test_a_record_cannot_supersede_itself() -> None:
    with pytest.raises(RecordError, match="cannot supersede itself"):
        make_record("R1", supersedes="R1")


@pytest.mark.parametrize("key", ["run_id", "cell_id", "status", "started_at", "finished_at"])
@pytest.mark.parametrize("bad", [None, {}, [], 7, "", "  "])
def test_required_fields_must_be_non_empty_strings(key: str, bad: Any) -> None:
    data = make_record("R1").to_dict()
    data[key] = bad

    with pytest.raises(RecordError, match=f"{key} must be a non-empty string"):
        RunRecord.from_dict(data)


@pytest.mark.parametrize("bad", [{}, [], 3])
def test_seed_role_must_be_a_string_when_present(bad: Any) -> None:
    data = make_record("R1").to_dict()
    data["seed_role"] = bad

    with pytest.raises(RecordError, match="seed_role must be a string"):
        RunRecord.from_dict(data)


def test_null_run_id_cannot_become_a_record_named_none() -> None:
    data = {
        "run_id": None,
        "cell_id": {},
        "seed": 0,
        "status": [],
        "started_at": "t",
        "finished_at": "t",
    }

    with pytest.raises(RecordError):
        RunRecord.from_dict(data)


def test_lenient_loading_skips_undecodable_records(tmp_path: Path) -> None:
    store = RegistryStore(tmp_path)
    store.write(make_record("R1"))
    (tmp_path / "R2.json").write_bytes(b'{"run_id": "R2", "cell_id": "\xff\xfe"}')

    assert [r.run_id for r in store.load_all()] == ["R1"]
    with pytest.raises(RecordError, match=r"R2\.json"):
        store.load_all(strict=True)


def test_distinct_cells_sharing_an_id_are_an_error() -> None:
    design = {"axes": {"detector": [f"d{i}" for i in range(40)]}}

    with pytest.raises(ExperimentsError, match=r"raise experiments\.cell_id_hash_length"):
        expand_design(design, prefix="C-", length=1)


def test_rules_that_map_combinations_to_one_cell_still_deduplicate() -> None:
    design = {
        "axes": {"detector": ["a"], "precision": ["fp32", "int8"]},
        "rules": [{"when": {"detector": "a"}, "fix": {"precision": "fp32"}}],
    }

    cells = expand_design(design, prefix="C-", length=10)

    assert [dict(c.factors) for c in cells] == [{"detector": "a", "precision": "fp32"}]


def test_allowed_signers_is_not_exempt_from_secret_scanning() -> None:
    config = tomllib.loads((REPO_ROOT / ".gitleaks.toml").read_text())
    paths = [p for rule in config.get("allowlists", []) for p in rule.get("paths", [])]

    assert not any("allowed_signers" in p for p in paths)
    assert json.dumps(config).count("allowed_signers") == 0
