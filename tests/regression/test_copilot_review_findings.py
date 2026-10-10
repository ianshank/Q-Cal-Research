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


# --- second Copilot review --------------------------------------------------------------------


def _commit_record(repo: Path, record: RunRecord) -> str:
    from tests.conftest import run_git, write

    write(repo, f"runs/registry/{record.run_id}.json", json.dumps(record.to_dict()))
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", f"add {record.run_id}")
    return run_git(repo, "rev-parse", "HEAD")


REASON = {"supersede_reason": "rerun"}


def test_supersede_cycles_are_rejected_by_registry_immutable(git_repo: Path) -> None:
    from qcal.ci.immutability import check_registry_immutable
    from tests.conftest import run_git

    base = run_git(git_repo, "rev-parse", "HEAD")
    _commit_record(git_repo, make_record("R2", supersedes="R3", provenance=REASON))
    head = _commit_record(git_repo, make_record("R3", supersedes="R2", provenance=REASON))

    violations = check_registry_immutable(git_repo, base, head).violations

    assert any("supersede cycle R2 -> R3 -> R2" in v for v in violations), violations


def test_supersede_cycles_fail_the_audit(config) -> None:
    from qcal.registry.audit import bad_supersedes

    records = [
        make_record("R2", supersedes="R3"),
        make_record("R3", supersedes="R4"),
        make_record("R4", supersedes="R2"),
        make_record("R5", supersedes="R4"),  # points into the cycle; not itself a cycle
    ]

    problems = bad_supersedes(records)

    assert "supersede cycle R2 -> R3 -> R4 -> R2" in problems
    assert not any(p.startswith("supersede cycle R5") for p in problems)


def test_registry_immutable_requires_the_supersede_reason(git_repo: Path) -> None:
    from qcal.ci.immutability import check_registry_immutable
    from tests.conftest import run_git

    base = run_git(git_repo, "rev-parse", "HEAD")
    _commit_record(git_repo, make_record("R1"))
    head = _commit_record(git_repo, make_record("R2", supersedes="R1"))

    assert check_registry_immutable(git_repo, base, head).violations == [
        "runs/registry/R2.json: supersedes 'R1' without provenance.supersede_reason"
    ]


def test_supersede_reason_check_follows_the_base_policy(git_repo: Path) -> None:
    from qcal.ci.immutability import check_registry_immutable
    from tests.conftest import run_git, write

    write(git_repo, "qcal.toml", "[registry]\nrequire_supersede_reason = false\n")
    run_git(git_repo, "commit", "-q", "-am", "policy without reasons")
    base = run_git(git_repo, "rev-parse", "HEAD")
    _commit_record(git_repo, make_record("R1"))
    head = _commit_record(git_repo, make_record("R2", supersedes="R1"))

    assert check_registry_immutable(git_repo, base, head).passed


@pytest.mark.parametrize("ref", ["agg:mean:AP:R1+R1+R2", "agg:max:AP:R2+R1+R2"])
def test_aggregate_references_name_each_run_once(config, ref: str) -> None:
    from qcal.integrity.claims import check_claims
    from qcal.registry.index import write_index
    from tests.conftest import write

    store = RegistryStore(config.path("registry_dir"))
    store.write(make_record("R1", seed=0, metrics={"AP": 40.0}))
    store.write(make_record("R2", seed=1, metrics={"AP": 50.0}))
    write_index(config, store)
    write(config.root, "paper/sections/results.tex", rf"\qcalval{{{ref}}}{{43.33}}" + "\n")

    messages = [f.message for f in check_claims(config)]

    assert any("more than once" in m for m in messages), messages


@pytest.mark.parametrize("bad", [[1, 2], {"a": 1}])
def test_design_axis_values_must_be_scalars(bad: Any) -> None:
    with pytest.raises(ExperimentsError, match="values must be scalars"):
        expand_design({"axes": {"opts": [bad]}}, prefix="C-", length=10)


def test_rule_fix_values_must_be_scalars() -> None:
    design = {
        "axes": {"detector": ["a"]},
        "rules": [{"when": {"detector": "a"}, "fix": {"opts": [1, 2]}}],
    }

    with pytest.raises(ExperimentsError, match="'fix' values must be scalars"):
        expand_design(design, prefix="C-", length=10)


@pytest.mark.parametrize("bad", [{"name": "draw"}, ["a"], 3])
def test_seed_role_must_be_a_string_in_the_pre_registration(config, bad: Any) -> None:
    from qcal.registry.experiments import parse_experiments

    data = {"version": 2, "seeds": [0], "seed_role": bad, "cells": [{"id": "C-a"}]}

    with pytest.raises(ExperimentsError, match="seed_role must be a string"):
        parse_experiments(config, data, path=Path("EXPERIMENTS.yaml"))


# --- third Copilot review (bfe4c4d) ----------------------------------------------------------


@pytest.mark.parametrize(
    "design",
    [
        {"axes": {"id": ["a", "b"], "detector": ["x"]}},
        {"axes": {"seeds": [0, 1]}},
        {"axes": {"detector": ["x"]}, "rules": [{"when": {"detector": "x"}, "fix": {"id": "y"}}]},
    ],
)
def test_reserved_keys_cannot_be_factor_names(design: dict[str, Any]) -> None:
    with pytest.raises(ExperimentsError, match="reserved key"):
        expand_design(design, prefix="C-", length=10, reserved=["id", "seeds"])


def test_an_emitted_cell_never_loses_its_id_to_a_factor() -> None:
    from qcal.registry.cells import cells_to_yaml_entries
    from qcal.registry.experiments import Cell

    with pytest.raises(ExperimentsError, match="reserved key"):
        cells_to_yaml_entries([Cell("C-1", {"id": "factor", "d": "a"})], "id")


def test_the_cells_command_refuses_a_reserved_factor(config) -> None:
    import io

    import yaml

    from qcal import cli
    from tests.conftest import write

    write(
        config.root,
        "EXPERIMENTS.yaml",
        yaml.safe_dump({"version": 2, "seeds": [0], "design": {"axes": {"id": ["a"]}}}),
    )
    out = io.StringIO()
    code = cli.main(["--root", str(config.root), "registry", "cells", "--emit"], out=out)
    assert code != 0
    assert "id:" not in out.getvalue()


# --- fourth Copilot review (4b1e597) ---------------------------------------------------------


def test_splits_verify_fails_on_missing_and_duplicated_manifests(repo: Path) -> None:
    import io
    import json

    from qcal.config import load_config
    from qcal_lab import cli
    from qcal_lab.data.splits import manifest_path
    from tests.conftest import write
    from tests.lab_support import coco_doc

    def verify() -> tuple[int, str]:
        out = io.StringIO()
        code = cli.main(
            ["--root", str(repo), "splits", "verify", "--annotations", str(ann)], out=out
        )
        return code, out.getvalue()

    ann = write(repo, "ann.json", json.dumps(coco_doc(images=10)))
    sizes = "trt_calib_images=1,calibrator_fit_split=3,val=3,test=3"
    partition = ["--root", str(repo), "splits", "partition", "--annotations", str(ann)]
    assert cli.main([*partition, "--seed", "1", "--sizes", sizes], out=io.StringIO()) == 0
    code, text = verify()
    assert (code, text.splitlines()[-1]) == (cli.EXIT_OK, "leakage: PASS"), text

    config = load_config(repo, environ={})
    test_manifest = manifest_path(config, "test")
    original = test_manifest.read_text()
    first = next(line for line in original.splitlines() if not line.startswith("#"))
    test_manifest.write_text(original + first + "\n")
    code, text = verify()
    assert code == cli.EXIT_FAILED
    assert "duplicate ids in test: 1" in text

    test_manifest.write_text(original)
    assert verify()[0] == cli.EXIT_OK
    manifest_path(config, "trt_calib_images").unlink()
    code, text = verify()  # no overlap and no unknown id, but a split is missing
    assert code == cli.EXIT_FAILED
    assert "trt_calib_images: no manifest" in text
    assert text.splitlines()[-1] == "leakage: FAIL"


def test_parity_readiness_needs_every_kind_of_oracle_case(repo: Path) -> None:
    import json

    from qcal.config import load_config
    from qcal_lab.config import load_lab_config
    from qcal_lab.status import build_status

    fixtures = repo / "tests/parity/fixtures"
    fixtures.mkdir(parents=True)
    oracle = {"repository": "fiveai/detection_calibration", "commit": "abc123"}

    def parity_item() -> tuple[bool, str]:
        report = build_status(load_config(repo, environ={}), load_lab_config(repo))
        return next(
            (ok, detail) for name, ok, detail in report.items if name == "oracle parity cases"
        )

    (fixtures / "iso.json").write_text(json.dumps({"kind": "calibrator", "oracle": oracle}))
    ok, detail = parity_item()
    assert not ok
    assert detail == "1 calibrator, 0 metric; none of kind metric"
    (fixtures / "laece.json").write_text(json.dumps({"kind": "metric", "oracle": oracle}))
    assert parity_item() == (True, "1 calibrator, 1 metric")
