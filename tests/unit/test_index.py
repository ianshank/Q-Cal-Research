"""The derived run index: rendering, column order, write/check, read-back and Parquet export."""

from __future__ import annotations

import csv
import importlib.util
import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qcal.config import Config, ConfigError, load_config
from qcal.registry.index import (
    IndexResult,
    columns_for,
    read_index,
    render_csv,
    render_value,
    superseded_ids,
    write_index,
)
from qcal.registry.records import RecordError, RunRecord
from qcal.registry.store import RegistryStore
from tests.conftest import make_record

PREFIXES = {"factors": "factor.", "metrics": "metric.", "environment": "env."}
CORE = [
    "run_id",
    "cell_id",
    "seed",
    "seed_role",
    "status",
    "supersedes",
    "started_at",
    "finished_at",
    "duration_s",
    "git_sha",
    "git_dirty",
    "config_hash",
]
finite_floats = st.floats(allow_nan=False, allow_infinity=False)


def store_for(config: Config, *records: RunRecord) -> RegistryStore:
    store = RegistryStore(config.path("registry_dir"))
    for record in records:
        store.write(record)
    return store


def parse_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


@pytest.fixture
def no_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``importlib.util.find_spec`` report that pyarrow is not installed."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> Any:
        return None if name.partition(".")[0] == "pyarrow" else real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


@pytest.fixture
def pyarrow_parquet() -> Any:
    return pytest.importorskip("pyarrow.parquet")


@pytest.fixture
def parquet_config(make_config: Callable[[str], Config]) -> Config:
    return make_config('[registry]\nwrite_parquet = "auto"\n')


# -- rendering values ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (-3, "-3"),
        (41.0, "41.0"),
        (0.1, "0.1"),
        (1e-300, "1e-300"),
        (1e16, "1e+16"),
        ("atss", "atss"),
        ("", ""),
        ({"b": 1, "a": [2, None]}, '{"a": [2, null], "b": 1}'),
        ([1, "x"], '[1, "x"]'),
        ((1, 2), "[1, 2]"),
        (Path("runs/x"), "runs/x"),
    ],
)
def test_render_value(value: Any, expected: str) -> None:
    assert render_value(value) == expected


@settings(database=None)
@given(value=st.floats(allow_nan=False))
def test_render_value_round_trips_floats_exactly(value: float) -> None:
    assert repr(float(render_value(value))) == repr(value)


# -- columns ------------------------------------------------------------------------------


def test_columns_start_with_core_columns_even_when_absent() -> None:
    assert columns_for([{"run_id": "R1"}], ["run_id", "status"], PREFIXES) == ["run_id", "status"]


def test_columns_order_groups_then_names() -> None:
    row = {
        "run_id": "R1",
        "zeta": 1,
        "env.python": "3.11",
        "metric.AP": 1.0,
        "factor.precision": "fp32",
        "metric.LaECE0": 2.0,
        "factor.detector": "atss",
        "alpha": 0,
    }
    assert columns_for([row], ["run_id"], PREFIXES) == [
        "run_id",
        "factor.detector",
        "factor.precision",
        "metric.AP",
        "metric.LaECE0",
        "env.python",
        "alpha",
        "zeta",
    ]


def test_columns_are_the_union_over_rows() -> None:
    rows = [{"run_id": "R1", "metric.AP": 1.0}, {"run_id": "R2", "metric.LRP": 2.0}]
    assert columns_for(rows, ["run_id"], PREFIXES) == ["run_id", "metric.AP", "metric.LRP"]


def test_columns_respect_custom_prefixes() -> None:
    prefixes = {"factors": "f:", "metrics": "m:", "environment": "e:"}
    row = {"e:x": 1, "m:x": 1, "f:x": 1}
    assert columns_for([row], [], prefixes) == ["f:x", "m:x", "e:x"]


def test_columns_fall_back_to_group_name_prefixes() -> None:
    row = {"environment.x": 1, "metrics.x": 1, "factors.x": 1}
    assert columns_for([row], [], {}) == ["factors.x", "metrics.x", "environment.x"]


# -- rendering the CSV --------------------------------------------------------------------


def test_render_csv_header_is_core_then_prefixed_columns(config: Config) -> None:
    text, _ = render_csv([make_record("R1")], config)
    header = text.splitlines()[0].split(",")
    assert header == [
        *CORE,
        "factor.detector",
        "factor.precision",
        "metric.AP",
        "metric.LaECE0",
    ]


def test_render_csv_sorts_rows_by_run_id(config: Config) -> None:
    text, count = render_csv([make_record("R3"), make_record("R1"), make_record("R2")], config)
    assert count == 3
    assert [row["run_id"] for row in parse_csv(text)] == ["R1", "R2", "R3"]


def test_render_csv_is_independent_of_input_order(config: Config) -> None:
    records = [make_record(f"R{i}", seed=i, metrics={"AP": i / 3}) for i in range(5)]
    assert render_csv(records, config) == render_csv(list(reversed(records)), config)


def test_render_csv_uses_unix_newlines_and_a_trailing_newline(config: Config) -> None:
    text, _ = render_csv([make_record("R1")], config)
    assert "\r" not in text
    assert text.endswith("\n")
    assert text.count("\n") == 2


def test_render_csv_renders_cell_values(config: Config) -> None:
    record = make_record(
        "R1",
        provenance={"git_sha": "abc", "git_dirty": False},
        environment={"gpus": ["a", "b"]},
        metrics={"AP": 41.0},
    )
    (row,) = parse_csv(render_csv([record], config)[0])
    assert (row["git_sha"], row["git_dirty"], row["config_hash"], row["supersedes"]) == (
        "abc",
        "false",
        "",
        "",
    )
    assert (row["metric.AP"], row["env.gpus"], row["seed"]) == ("41.0", '["a", "b"]', "0")


def test_render_csv_leaves_missing_group_values_empty(config: Config) -> None:
    records = [make_record("R1", metrics={"AP": 1.0}), make_record("R2", metrics={"LRP": 2.0})]
    rows = parse_csv(render_csv(records, config)[0])
    assert [(r["metric.AP"], r["metric.LRP"]) for r in rows] == [("1.0", ""), ("", "2.0")]


def test_render_csv_quotes_values_that_need_it(config: Config) -> None:
    record = make_record("R1", factors={"note": 'a, "b"\nc'})
    (row,) = parse_csv(render_csv([record], config)[0])
    assert row["factor.note"] == 'a, "b"\nc'


def test_render_csv_of_no_records_is_just_the_header(config: Config) -> None:
    assert render_csv([], config) == (",".join(CORE) + "\n", 0)


def test_render_csv_uses_configured_core_columns_and_prefixes(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config(
        """
        [registry]
        core_columns = ["run_id", "status"]

        [registry.column_prefixes]
        factors = "f_"
        metrics = "m_"
        environment = "e_"
        """
    )
    text, _ = render_csv([make_record("R1", factors={"d": "x"}, metrics={"AP": 1.0})], config)
    assert text.splitlines()[0].split(",") == [
        "run_id",
        "status",
        "f_d",
        "m_AP",
        # flat fields that are not core columns still appear, after the groups, by name
        "cell_id",
        "config_hash",
        "duration_s",
        "finished_at",
        "git_dirty",
        "git_sha",
        "seed",
        "seed_role",
        "started_at",
        "supersedes",
    ]


# -- writing and checking -----------------------------------------------------------------


def test_write_index_writes_the_rendered_csv(config: Config) -> None:
    store = store_for(config, make_record("R1"), make_record("R2"))
    result = write_index(config, store)
    target = config.path("index_csv")
    assert (result.path, result.rows, result.changed) == (target, 2, True)
    assert target.read_text("utf-8") == render_csv(store.load_all(), config)[0]


def test_write_index_reports_unchanged_on_second_run(config: Config) -> None:
    store = store_for(config, make_record("R1"))
    write_index(config, store)
    before = config.path("index_csv").read_text("utf-8")
    result = write_index(config, store)
    assert result.changed is False
    assert config.path("index_csv").read_text("utf-8") == before


def test_write_index_picks_up_new_records(config: Config) -> None:
    store = store_for(config, make_record("R1"))
    write_index(config, store)
    store.write(make_record("R2"))
    result = write_index(config, store)
    assert (result.changed, result.rows) == (True, 2)
    assert [r["run_id"] for r in read_index(config.path("index_csv"))] == ["R1", "R2"]


def test_empty_registry_without_index_is_consistent(config: Config) -> None:
    result = write_index(config, store_for(config))
    assert (result.rows, result.changed, result.parquet) == (0, False, None)
    assert not config.path("index_csv").exists()


def test_empty_registry_check_without_index_is_not_stale(config: Config) -> None:
    assert write_index(config, store_for(config), check=True).changed is False


def test_stale_index_with_no_records_is_rewritten(config: Config) -> None:
    target = config.path("index_csv")
    target.parent.mkdir(parents=True)
    target.write_text("run_id\nR-gone\n", "utf-8")
    result = write_index(config, store_for(config))
    assert result.changed is True
    assert target.read_text("utf-8") == ",".join(CORE) + "\n"


def test_check_reports_missing_index_without_writing(config: Config) -> None:
    result = write_index(config, store_for(config, make_record("R1")), check=True)
    assert result == IndexResult(config.path("index_csv"), 1, True)
    assert not config.path("index_csv").exists()


def test_check_reports_stale_index_without_touching_it(config: Config) -> None:
    store = store_for(config, make_record("R1"))
    write_index(config, store)
    store.write(make_record("R2"))
    before = config.path("index_csv").read_text("utf-8")
    assert write_index(config, store, check=True).changed is True
    assert config.path("index_csv").read_text("utf-8") == before


def test_check_reports_fresh_index_as_unchanged(config: Config) -> None:
    store = store_for(config, make_record("R1"))
    write_index(config, store)
    assert write_index(config, store, check=True).changed is False


def test_check_detects_hand_edited_index(config: Config) -> None:
    store = store_for(config, make_record("R1"))
    write_index(config, store)
    target = config.path("index_csv")
    target.write_text(target.read_text("utf-8").replace("41.0", "45.0"), "utf-8")
    assert write_index(config, store, check=True).changed is True


def test_write_index_refuses_malformed_records(config: Config) -> None:
    store = store_for(config, make_record("R1"))
    (store.directory / "R2.json").write_text("{ broken", "utf-8")
    with pytest.raises(RecordError, match=r"R2\.json"):
        write_index(config, store)


def test_write_index_uses_configured_path(make_config: Callable[[str], Config]) -> None:
    config = make_config('[paths]\nindex_csv = "out/runs.csv"\n')
    result = write_index(config, store_for(config, make_record("R1")))
    assert result.path == config.root / "out" / "runs.csv"
    assert result.path.is_file()


# -- reading ------------------------------------------------------------------------------


def test_read_index_of_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_index(tmp_path / "missing.csv") == []


def test_read_index_returns_string_rows(config: Config) -> None:
    write_index(config, store_for(config, make_record("R1", supersedes="R0")))
    (row,) = read_index(config.path("index_csv"))
    assert (row["run_id"], row["seed"], row["supersedes"], row["metric.AP"]) == (
        "R1",
        "0",
        "R0",
        "41.0",
    )


@settings(database=None, deadline=None, max_examples=40)
@given(values=st.lists(finite_floats, min_size=1, max_size=5))
def test_metric_floats_round_trip_through_the_index(
    tmp_path_factory: pytest.TempPathFactory, values: list[float]
) -> None:
    root = tmp_path_factory.mktemp("index")
    (root / "qcal.toml").write_text('[registry]\nwrite_parquet = "never"\n', "utf-8")
    config = load_config(root, environ={})
    records = [make_record(f"R{i:02d}", metrics={"m": v}) for i, v in enumerate(values)]
    write_index(config, store_for(config, *records))
    read_back = [float(row["metric.m"]) for row in read_index(config.path("index_csv"))]
    assert [repr(v) for v in read_back] == [repr(v) for v in values]


# -- superseded ids -----------------------------------------------------------------------


def test_superseded_ids_collects_superseded_run_ids() -> None:
    rows = [
        {"run_id": "R1", "supersedes": ""},
        {"run_id": "R2", "supersedes": "R1"},
        {"run_id": "R3"},
        {"run_id": "R4", "supersedes": "R2"},
    ]
    assert superseded_ids(rows) == frozenset({"R1", "R2"})


def test_superseded_ids_of_no_rows_is_empty() -> None:
    assert superseded_ids([]) == frozenset()


def test_superseded_ids_agrees_with_the_written_index(config: Config) -> None:
    write_index(config, store_for(config, make_record("R1"), make_record("R2", supersedes="R1")))
    assert superseded_ids(read_index(config.path("index_csv"))) == {"R1"}


# -- Parquet ------------------------------------------------------------------------------


def test_parquet_is_written_when_pyarrow_is_available(
    parquet_config: Config, pyarrow_parquet: Any
) -> None:
    result = write_index(parquet_config, store_for(parquet_config, make_record("R1")))
    assert result.parquet == parquet_config.path("index_parquet")
    table = pyarrow_parquet.read_table(result.parquet)
    assert table.num_rows == 1
    assert table.column_names[:2] == ["run_id", "cell_id"]


def test_parquet_always_mode_writes_when_pyarrow_is_available(
    make_config: Callable[[str], Config], pyarrow_parquet: Any
) -> None:
    config = make_config('[registry]\nwrite_parquet = "always"\n')
    result = write_index(config, store_for(config, make_record("R1")))
    assert result.parquet is not None
    assert result.parquet.is_file()


def test_parquet_never_mode_skips_the_export(make_config: Callable[[str], Config]) -> None:
    config = make_config('[registry]\nwrite_parquet = "never"\n')
    result = write_index(config, store_for(config, make_record("R1")))
    assert result.parquet is None
    assert not config.path("index_parquet").exists()


def test_parquet_auto_mode_skips_without_pyarrow(parquet_config: Config, no_pyarrow: None) -> None:
    result = write_index(parquet_config, store_for(parquet_config, make_record("R1")))
    assert result.parquet is None
    assert result.changed is True
    assert not parquet_config.path("index_parquet").exists()


def test_parquet_always_mode_requires_pyarrow(
    make_config: Callable[[str], Config], no_pyarrow: None
) -> None:
    config = make_config('[registry]\nwrite_parquet = "always"\n')
    with pytest.raises(ConfigError, match="pyarrow is not installed"):
        write_index(config, store_for(config, make_record("R1")))


@pytest.mark.parametrize("mode", ["yes", "", "Auto"])
def test_parquet_mode_must_be_known(make_config: Callable[[str], Config], mode: str) -> None:
    config = make_config(f'[registry]\nwrite_parquet = "{mode}"\n')
    with pytest.raises(ConfigError, match="must be auto, always or never"):
        write_index(config, store_for(config, make_record("R1")))


def test_check_mode_never_writes_parquet(parquet_config: Config, pyarrow_parquet: Any) -> None:
    result = write_index(parquet_config, store_for(parquet_config, make_record("R1")), check=True)
    assert result.parquet is None
    assert not parquet_config.path("index_parquet").exists()


def test_parquet_newer_than_unchanged_csv_is_not_rewritten(
    parquet_config: Config, pyarrow_parquet: Any
) -> None:
    store = store_for(parquet_config, make_record("R1"))
    parquet = write_index(parquet_config, store).parquet
    assert parquet is not None
    sentinel = parquet_config.path("index_csv").stat().st_mtime + 1000
    os.utime(parquet, (sentinel, sentinel))
    result = write_index(parquet_config, store)
    assert (result.changed, result.parquet) == (False, parquet)
    assert parquet.stat().st_mtime == sentinel


def test_parquet_older_than_csv_is_rewritten(parquet_config: Config, pyarrow_parquet: Any) -> None:
    store = store_for(parquet_config, make_record("R1"))
    parquet = write_index(parquet_config, store).parquet
    assert parquet is not None
    old = parquet_config.path("index_csv").stat().st_mtime - 1000
    os.utime(parquet, (old, old))
    write_index(parquet_config, store)
    assert parquet.stat().st_mtime > old


def test_missing_parquet_is_restored_for_an_unchanged_csv(
    parquet_config: Config, pyarrow_parquet: Any
) -> None:
    store = store_for(parquet_config, make_record("R1"))
    parquet = write_index(parquet_config, store).parquet
    assert parquet is not None
    parquet.unlink()
    result = write_index(parquet_config, store)
    assert (result.changed, result.parquet) == (False, parquet)
    assert pyarrow_parquet.read_table(parquet).num_rows == 1


def test_parquet_is_not_written_for_an_empty_registry(
    parquet_config: Config, pyarrow_parquet: Any
) -> None:
    assert write_index(parquet_config, store_for(parquet_config)).parquet is None
    assert not parquet_config.path("index_parquet").exists()


def test_parquet_follows_a_changed_csv_regardless_of_timestamps(
    parquet_config: Config, pyarrow_parquet: Any
) -> None:
    store = store_for(parquet_config, make_record("R1"))
    parquet = write_index(parquet_config, store).parquet
    assert parquet is not None
    # Coarse filesystem timestamps (or clock skew) make the old export look up to date.
    ahead = parquet.stat().st_mtime + 1000
    os.utime(parquet, (ahead, ahead))
    store.write(make_record("R2"))
    assert write_index(parquet_config, store).changed is True
    assert pyarrow_parquet.read_table(parquet).num_rows == 2
