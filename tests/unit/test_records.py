"""Run records: run-id validation, artifact refs, (de)serialisation, flattening, supersession."""

from __future__ import annotations

import math
import re
from typing import Any

import pytest

from qcal.registry.records import (
    SCHEMA_VERSION,
    ArtifactRef,
    RecordError,
    RunRecord,
    effective,
    validate_run_id,
)
from tests.conftest import make_record

PREFIXES = {"factors": "factor.", "metrics": "metric.", "environment": "env."}
REQUIRED = ("run_id", "cell_id", "seed", "status", "started_at", "finished_at")


def minimal_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run_id": "R1",
        "cell_id": "C-a",
        "seed": 0,
        "status": "ok",
        "started_at": "2026-10-09T00:00:00+00:00",
        "finished_at": "2026-10-09T00:01:00+00:00",
    }
    data.update(overrides)
    return data


# -- run ids ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "run_id",
    ["R1", "a", "0", "R20261009T120000Z-C-0a1b2c3d4e-s0-a1b2c3", "x.y_z-1", "A.B.C"],
)
def test_validate_run_id_returns_valid_ids_unchanged(run_id: str) -> None:
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "-R1",
        ".hidden",
        "_R1",
        "../escape",
        "a/b",
        "a\\b",
        "has space",
        "R1\n",
        "run:R1",
        "R1+R2",
        "café",
    ],
)
def test_validate_run_id_rejects_unsafe_ids(run_id: str) -> None:
    with pytest.raises(RecordError, match="must match"):
        validate_run_id(run_id)


# -- artifact refs ------------------------------------------------------------------------


def test_artifact_ref_round_trips_through_dict() -> None:
    ref = ArtifactRef("runs/results/x.engine", "ab" * 32, "engine", 1234)
    assert ArtifactRef.from_dict(ref.to_dict()) == ref


def test_artifact_ref_to_dict_has_every_field() -> None:
    ref = ArtifactRef("p", "h")
    assert ref.to_dict() == {"path": "p", "sha256": "h", "kind": "", "size": None}


def test_artifact_ref_from_dict_defaults_kind_and_size() -> None:
    ref = ArtifactRef.from_dict({"path": "p", "sha256": "h"})
    assert (ref.kind, ref.size) == ("", None)


def test_artifact_ref_from_dict_drops_non_integer_size() -> None:
    assert ArtifactRef.from_dict({"path": "p", "sha256": "h", "size": "12"}).size is None


@pytest.mark.parametrize(
    "data",
    [{}, {"path": "p"}, {"sha256": "h"}, {"path": 1, "sha256": "h"}, {"path": "p", "sha256": None}],
)
def test_artifact_ref_from_dict_requires_string_path_and_hash(data: dict[str, Any]) -> None:
    with pytest.raises(RecordError, match="string 'path' and 'sha256'"):
        ArtifactRef.from_dict(data)


# -- construction -------------------------------------------------------------------------


def test_record_rejects_invalid_run_id() -> None:
    with pytest.raises(RecordError, match="must match"):
        make_record("../R1")


def test_record_rejects_invalid_supersedes_id() -> None:
    with pytest.raises(RecordError, match="must match"):
        make_record("R2", supersedes="not/valid")


@pytest.mark.parametrize("value", [True, False, "1.0", None, [1.0], math.nan, math.inf, -math.inf])
def test_record_rejects_non_finite_or_non_numeric_metrics(value: Any) -> None:
    with pytest.raises(RecordError, match="must be a finite number"):
        make_record(metrics={"AP": value})


@pytest.mark.parametrize(
    "name",
    [
        "a:b",
        "a+b",
        "a b",
        "a,b",
        "{a}",
        "",
        "AP\n",
        "-AP",
        ".AP",
        "@AP",
        "a/b",
        "a\\b",
        "AP$_0$",
        "café",
    ],
)
def test_record_rejects_metric_names_unusable_in_claim_references(name: str) -> None:
    with pytest.raises(RecordError, match="metric name"):
        make_record(metrics={name: 1.0})


@pytest.mark.parametrize(
    "name", ["AP", "LaECE0", "AP@50", "latency_p50_ms", "AP.small", "brier-iou", "_x", "0"]
)
def test_record_accepts_metric_names_matching_the_documented_pattern(name: str) -> None:
    assert list(make_record(metrics={name: 1.0}).metrics) == [name]


def test_record_accepts_integer_and_float_metrics() -> None:
    record = make_record(metrics={"count": 3, "AP": 41.5, "neg": -1e-12})
    assert dict(record.metrics) == {"count": 3, "AP": 41.5, "neg": -1e-12}


def test_record_is_immutable() -> None:
    record = make_record()
    with pytest.raises(AttributeError, match="cannot assign"):
        record.status = "failed"  # type: ignore[misc]


# -- serialisation ------------------------------------------------------------------------


def test_to_dict_lists_every_documented_field() -> None:
    data = make_record().to_dict()
    assert set(data) == {
        "schema_version",
        "run_id",
        "cell_id",
        "seed",
        "seed_role",
        "status",
        "supersedes",
        "started_at",
        "finished_at",
        "duration_s",
        "provenance",
        "factors",
        "metrics",
        "environment",
        "artifacts",
        "log_path",
        "error",
    }


def test_to_dict_serialises_artifacts_as_dicts() -> None:
    record = make_record(artifacts=(ArtifactRef("a.bin", "h", "engine", 3),))
    assert record.to_dict()["artifacts"] == [
        {"path": "a.bin", "sha256": "h", "kind": "engine", "size": 3}
    ]


def test_to_dict_merges_extra_keys() -> None:
    assert make_record(extra={"note": "x"}).to_dict()["note"] == "x"


def test_to_dict_extra_keys_never_override_core_fields() -> None:
    data = make_record("R1", extra={"run_id": "EVIL", "status": "ok?"}).to_dict()
    assert (data["run_id"], data["status"]) == ("R1", "ok")


def test_record_round_trips_through_dict() -> None:
    record = make_record(
        "R2",
        supersedes="R1",
        duration_s=60.0,
        provenance={"git_sha": "abc", "git_dirty": False, "config_hash": "h"},
        environment={"python": "3.11"},
        artifacts=(ArtifactRef("x.engine", "h" * 64, "engine", 10),),
        log_path="runs/logs/R2.log",
        error=None,
        schema_version=1,
        extra={"note": "kept"},
    )
    assert RunRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize("missing", REQUIRED)
def test_from_dict_requires_core_fields(missing: str) -> None:
    data = minimal_dict()
    del data[missing]
    with pytest.raises(RecordError, match=f"missing {missing}"):
        RunRecord.from_dict(data)


def test_from_dict_reports_every_missing_field() -> None:
    with pytest.raises(RecordError, match="missing run_id, cell_id, seed"):
        RunRecord.from_dict({"status": "ok", "started_at": "a", "finished_at": "b"})


@pytest.mark.parametrize("seed", ["0", 1.0, True, None])
def test_from_dict_requires_integer_seed(seed: Any) -> None:
    with pytest.raises(RecordError, match="seed must be an integer"):
        RunRecord.from_dict(minimal_dict(seed=seed))


@pytest.mark.parametrize("key", ["provenance", "factors", "metrics", "environment"])
@pytest.mark.parametrize("value", [[], "x", None, 3])
def test_from_dict_requires_object_groups(key: str, value: Any) -> None:
    with pytest.raises(RecordError, match=f"{key} must be an object"):
        RunRecord.from_dict(minimal_dict(**{key: value}))


@pytest.mark.parametrize("value", [{}, "x", None])
def test_from_dict_requires_artifact_list(value: Any) -> None:
    with pytest.raises(RecordError, match="artifacts must be a list"):
        RunRecord.from_dict(minimal_dict(artifacts=value))


def test_from_dict_validates_metrics() -> None:
    with pytest.raises(RecordError, match="finite number"):
        RunRecord.from_dict(minimal_dict(metrics={"AP": "high"}))


def test_from_dict_applies_defaults_for_optional_fields() -> None:
    record = RunRecord.from_dict(minimal_dict())
    assert (
        record.seed_role,
        record.supersedes,
        record.duration_s,
        record.artifacts,
        record.schema_version,
        dict(record.extra),
    ) == ("", None, None, (), 1, {})


@pytest.mark.parametrize(("raw", "expected"), [(5, 5.0), (2.5, 2.5), ("5", None), (None, None)])
def test_from_dict_normalises_duration(raw: Any, expected: float | None) -> None:
    assert RunRecord.from_dict(minimal_dict(duration_s=raw)).duration_s == expected


def test_from_dict_treats_empty_supersedes_as_none() -> None:
    assert RunRecord.from_dict(minimal_dict(supersedes="")).supersedes is None


def test_from_dict_coerces_null_seed_role_to_empty_string() -> None:
    assert RunRecord.from_dict(minimal_dict(seed_role=None)).seed_role == ""


def test_from_dict_keeps_unknown_keys_as_extra() -> None:
    record = RunRecord.from_dict(minimal_dict(experiment_id="E1", target="trt_jetson"))
    assert dict(record.extra) == {"experiment_id": "E1", "target": "trt_jetson"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"supersedes": 7}, "supersedes must be a run id string or null"),
        ({"supersedes": ["R0"]}, "supersedes must be a run id string or null"),
        ({"schema_version": "v1"}, "schema_version must be an integer"),
        ({"schema_version": None}, "schema_version must be an integer"),
        ({"schema_version": True}, "schema_version must be an integer"),
        ({"artifacts": ["x"]}, "artifacts must be a list of objects"),
        ({"artifacts": [{"path": 1, "sha256": "h"}]}, "string 'path' and 'sha256'"),
        ({"supersedes": "not/valid"}, "must match"),
    ],
    ids=[
        "int-supersedes",
        "list-supersedes",
        "str-schema-version",
        "null-schema-version",
        "bool-schema-version",
        "non-object-artifact",
        "bad-artifact-path",
        "unsafe-supersedes",
    ],
)
def test_from_dict_reports_mistyped_fields_as_record_errors(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(RecordError, match=re.escape(message)):
        RunRecord.from_dict(minimal_dict(**overrides))


@pytest.mark.parametrize("version", [0, -1, SCHEMA_VERSION + 1, 99])
def test_from_dict_refuses_schema_versions_this_code_cannot_read(version: int) -> None:
    """A newer schema may mean what old code would misread, so old code refuses it."""
    with pytest.raises(RecordError, match=f"schema_version {version} is not readable"):
        RunRecord.from_dict(minimal_dict(schema_version=version))


def test_from_dict_reads_records_without_a_schema_version_as_version_1() -> None:
    data = minimal_dict()
    data.pop("schema_version", None)
    assert RunRecord.from_dict(data).schema_version == 1


def test_from_dict_wraps_unexpected_type_errors() -> None:
    not_a_mapping: Any = list(REQUIRED)  # every required key is "in" it, but it is not a dict
    with pytest.raises(RecordError, match="malformed record"):
        RunRecord.from_dict(not_a_mapping)


# -- flattening ---------------------------------------------------------------------------


def test_flat_contains_core_and_provenance_columns() -> None:
    record = make_record(
        provenance={"git_sha": "abc", "git_dirty": True, "config_hash": "h", "other": "x"}
    )
    row = record.flat(PREFIXES)
    assert {k: row[k] for k in ("run_id", "git_sha", "git_dirty", "config_hash")} == {
        "run_id": "R1",
        "git_sha": "abc",
        "git_dirty": True,
        "config_hash": "h",
    }


def test_flat_omits_provenance_keys_that_are_not_columns() -> None:
    row = make_record(provenance={"executor": "FakeExecutor"}).flat(PREFIXES)
    assert "executor" not in row
    assert row["git_sha"] is None


def test_flat_prefixes_nested_groups() -> None:
    row = make_record(environment={"python": "3.11"}).flat(PREFIXES)
    assert {k: v for k, v in row.items() if "." in k} == {
        "factor.detector": "atss",
        "factor.precision": "fp32",
        "metric.LaECE0": 12.5,
        "metric.AP": 41.0,
        "env.python": "3.11",
    }


def test_flat_uses_custom_prefixes() -> None:
    row = make_record(factors={"d": "x"}, metrics={"m": 1.0}).flat(
        {"factors": "f_", "metrics": "m_", "environment": "e_"}
    )
    assert (row["f_d"], row["m_m"]) == ("x", 1.0)


def test_flat_falls_back_to_group_name_prefix() -> None:
    row = make_record(factors={"d": "x"}, metrics={"m": 1.0}).flat({})
    assert (row["factors.d"], row["metrics.m"]) == ("x", 1.0)


# -- supersession -------------------------------------------------------------------------


def test_effective_keeps_records_without_supersession() -> None:
    records = [make_record("R1"), make_record("R2")]
    assert effective(records) == records


def test_effective_drops_superseded_records() -> None:
    records = [make_record("R1"), make_record("R2", supersedes="R1"), make_record("R3")]
    assert [r.run_id for r in effective(records)] == ["R2", "R3"]


def test_effective_follows_supersession_chains() -> None:
    records = [
        make_record("R1"),
        make_record("R2", supersedes="R1"),
        make_record("R3", supersedes="R2"),
    ]
    assert [r.run_id for r in effective(records)] == ["R3"]


def test_effective_ignores_order_of_records() -> None:
    records = [make_record("R2", supersedes="R1"), make_record("R1")]
    assert [r.run_id for r in effective(records)] == ["R2"]


def test_effective_ignores_supersession_of_unknown_run() -> None:
    records = [make_record("R1"), make_record("R2", supersedes="R0")]
    assert [r.run_id for r in effective(records)] == ["R1", "R2"]


def test_failure_kind_round_trips_and_is_absent_when_unset() -> None:
    record = make_record(status="failed", failure_kind="timeout")
    assert RunRecord.from_dict(record.to_dict()).failure_kind == "timeout"
    assert "failure_kind" not in make_record().to_dict()
    assert RunRecord.from_dict(make_record().to_dict()).failure_kind is None


def test_failure_kind_must_be_a_string() -> None:
    with pytest.raises(RecordError, match="failure_kind must be a string or null"):
        RunRecord.from_dict({**make_record().to_dict(), "failure_kind": 3})
