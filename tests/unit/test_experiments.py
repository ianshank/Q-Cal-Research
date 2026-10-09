"""Loading and validating the pre-registration (EXPERIMENTS.yaml)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import yaml

from qcal.config import Config
from qcal.registry.experiments import (
    Cell,
    Experiments,
    ExperimentsError,
    find_placeholders,
    load_experiments,
    parse_experiments,
)
from tests.conftest import experiments_yaml, write

CELLS: list[dict[str, Any]] = [
    {"id": "C-a", "detector": "atss", "precision": "fp32"},
    {"id": "C-b", "detector": "detr", "precision": "int8", "seeds": [7, 8]},
    {"id": "D-c", "detector": "yolox", "precision": "fp16"},
]


def parse(config: Config, data: dict[str, Any]) -> Experiments:
    return parse_experiments(config, data, path=config.path("experiments"))


@pytest.fixture
def experiments(config: Config, repo: Path) -> Experiments:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml(CELLS))
    return load_experiments(config)


# -- loading ------------------------------------------------------------------------------


def test_load_reports_missing_file(config: Config) -> None:
    with pytest.raises(ExperimentsError, match="does not exist"):
        load_experiments(config)


def test_load_reports_unparseable_yaml(config: Config, repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", "cells: [unclosed\n")
    with pytest.raises(ExperimentsError, match="cannot parse"):
        load_experiments(config)


@pytest.mark.parametrize("body", ["- a\n- b\n", "just a string\n", "42\n"])
def test_load_requires_a_top_level_mapping(config: Config, repo: Path, body: str) -> None:
    write(repo, "EXPERIMENTS.yaml", body)
    with pytest.raises(ExperimentsError, match="mapping at the top level"):
        load_experiments(config)


def test_load_of_empty_file_has_no_cells_or_seeds(config: Config, repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", "")
    loaded = load_experiments(config)
    assert (loaded.cells, loaded.seeds, loaded.placeholders) == ((), (), ())


def test_load_records_path_and_sha256_of_raw_bytes(config: Config, repo: Path) -> None:
    path = write(repo, "EXPERIMENTS.yaml", experiments_yaml(CELLS))
    loaded = load_experiments(config)
    assert loaded.path == config.path("experiments")
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_keeps_raw_data(experiments: Experiments) -> None:
    assert experiments.data["version"] == 2


def test_load_accepts_explicit_path(config: Config, tmp_path: Path) -> None:
    path = write(tmp_path, "elsewhere/E.yaml", experiments_yaml(CELLS[:1]))
    loaded = load_experiments(config, path)
    assert (loaded.path, [c.id for c in loaded.cells]) == (path, ["C-a"])


def test_load_uses_configured_path(make_config: Callable[[str], Config], repo: Path) -> None:
    config = make_config('[paths]\nexperiments = "prereg/E.yaml"\n')
    write(repo, "prereg/E.yaml", experiments_yaml(CELLS[:1]))
    assert [c.id for c in load_experiments(config).cells] == ["C-a"]


# -- seeds and roles ----------------------------------------------------------------------


def test_global_seeds_become_a_tuple(experiments: Experiments) -> None:
    assert experiments.seeds == (0, 1, 2)


def test_seed_role_comes_from_the_file(experiments: Experiments) -> None:
    assert experiments.seed_role == "calibrator_fit_draw"


def test_seed_role_defaults_from_configuration(config: Config) -> None:
    assert parse(config, {"seeds": [0]}).seed_role == "seed"


def test_missing_seeds_key_means_no_seeds(config: Config) -> None:
    assert parse(config, {}).seeds == ()


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ("0,1", "must be a list of integers"),
        ([0, "1"], "must be a list of integers"),
        ([0, 1.5], "must be a list of integers"),
        ([True], "must be a list of integers"),
        (None, "must be a list of integers"),
        ([0, 1, 0], "contains duplicates"),
    ],
)
def test_invalid_global_seeds_are_rejected(config: Config, seeds: Any, message: str) -> None:
    with pytest.raises(ExperimentsError, match=f"seeds {message}"):
        parse(config, {"seeds": seeds})


@pytest.mark.parametrize("seeds", [[1, 1], ["a"], 3])
def test_invalid_cell_seeds_name_the_cell(config: Config, seeds: Any) -> None:
    with pytest.raises(ExperimentsError, match="cell C-a seeds"):
        parse(config, {"seeds": [0], "cells": [{"id": "C-a", "seeds": seeds}]})


# -- cells --------------------------------------------------------------------------------


def test_cells_keep_file_order_and_factors(experiments: Experiments) -> None:
    assert [(c.id, dict(c.factors)) for c in experiments.cells] == [
        ("C-a", {"detector": "atss", "precision": "fp32"}),
        ("C-b", {"detector": "detr", "precision": "int8"}),
        ("D-c", {"detector": "yolox", "precision": "fp16"}),
    ]


def test_per_cell_seeds_are_parsed_and_excluded_from_factors(experiments: Experiments) -> None:
    cell = experiments.cell("C-b")
    assert cell.seeds == (7, 8)
    assert "seeds" not in cell.factors


def test_cells_without_seeds_have_none(experiments: Experiments) -> None:
    assert experiments.cell("C-a").seeds is None


@pytest.mark.parametrize("value", [None, True, 3, 2.5, "x"])
def test_scalar_factor_values_are_accepted(config: Config, value: Any) -> None:
    cells = parse(config, {"cells": [{"id": "C-a", "f": value}]}).cells
    assert cells[0].factors == {"f": value}


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        ({"id": "C-a"}, "cells must be a list"),
        (["C-a"], r"cells\[0\] must be a mapping"),
        ([{"detector": "atss"}], r"cells\[0\] needs a string 'id'"),
        ([{"id": ""}], r"cells\[0\] needs a string 'id'"),
        ([{"id": 7}], r"cells\[0\] needs a string 'id'"),
        ([{"id": "C-a"}, {"id": "C-a"}], "duplicate cell id 'C-a'"),
        ([{"id": "C-a", "opts": [1, 2]}], r"factors must be scalars \(opts\)"),
        ([{"id": "C-a", "opts": {"k": 1}}], r"factors must be scalars \(opts\)"),
    ],
)
def test_malformed_cells_are_rejected(config: Config, cells: Any, message: str) -> None:
    with pytest.raises(ExperimentsError, match=message):
        parse(config, {"seeds": [0], "cells": cells})


def test_null_cells_means_no_cells(config: Config) -> None:
    assert parse(config, {"cells": None}).cells == ()


def test_keys_are_configurable(make_config: Callable[[str], Config]) -> None:
    config = make_config(
        """
        [experiments]
        cells_key = "grid"
        cell_id_key = "name"
        seeds_key = "draws"
        seed_role_key = "role"
        """
    )
    data = {"draws": [3], "role": "qat", "grid": [{"name": "G1", "draws": [4], "lr": 0.1}]}
    loaded = parse(config, data)
    assert (loaded.seeds, loaded.seed_role) == ((3,), "qat")
    assert loaded.cells == (Cell("G1", {"lr": 0.1}, (4,)),)


# -- lookups ------------------------------------------------------------------------------


def test_cell_lookup_returns_registered_cell(experiments: Experiments) -> None:
    assert experiments.cell("D-c").factors["detector"] == "yolox"


def test_cell_lookup_rejects_unregistered_cell(experiments: Experiments) -> None:
    with pytest.raises(ExperimentsError, match="'C-z' is not pre-registered in EXPERIMENTS.yaml"):
        experiments.cell("C-z")


def test_seeds_for_prefers_cell_seeds(experiments: Experiments) -> None:
    assert experiments.seeds_for(experiments.cell("C-b")) == (7, 8)


def test_seeds_for_falls_back_to_global_seeds(experiments: Experiments) -> None:
    assert experiments.seeds_for(experiments.cell("C-a")) == (0, 1, 2)


@pytest.mark.parametrize(
    ("patterns", "expected"),
    [
        ("C-*", ["C-a", "C-b"]),
        ("*", ["C-a", "C-b", "D-c"]),
        ("C-a", ["C-a"]),
        ("D-c,C-a", ["C-a", "D-c"]),
        (" C-a , D-? ", ["C-a", "D-c"]),
        (["C-b", "D-*"], ["C-b", "D-c"]),
        ("C-*,*", ["C-a", "C-b", "D-c"]),
        ("C-[ab]", ["C-a", "C-b"]),
        ("c-a", []),
        ("", []),
        (",,", []),
        ([], []),
        ("X-*", []),
    ],
)
def test_match_selects_cells_by_fnmatch_patterns(
    experiments: Experiments, patterns: str | list[str], expected: list[str]
) -> None:
    assert [c.id for c in experiments.match(patterns)] == expected


# -- placeholders -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"question": {"ian": "state it"}}, ["question.ian"]),
        ({"seed_role": "ian: name it"}, ["seed_role"]),
        ({"seed_role": "  ian: padded"}, ["seed_role"]),
        ({"a": {"b": ["x", "ian: todo"]}}, ["a.b[1]"]),
        ({"a": [{"ian": 1}]}, ["a[0].ian"]),
        ({"a": "see ian: later", "b": "ian", "c": "Ian: other"}, []),
        ({"a": 1, "b": None, "c": True}, []),
        ({1: "ian: numeric key"}, ["1"]),
    ],
)
def test_find_placeholders_reports_dotted_keys(data: Any, expected: list[str]) -> None:
    assert find_placeholders(data, "ian:") == expected


def test_find_placeholders_uses_the_given_marker() -> None:
    data = {"a": "TODO: x", "TODO": 1, "b": "ian: y"}
    assert find_placeholders(data, "TODO:") == ["a", "TODO"]


def test_find_placeholders_reports_each_location_once() -> None:
    assert find_placeholders({"q": {"ian": "ian: todo"}}, "ian:") == ["q.ian"]


def test_find_placeholders_does_not_descend_into_a_placeholder_key() -> None:
    data = {"q": {"ian": {"ian": "ian: nested", "items": ["ian: x"]}}}
    assert find_placeholders(data, "ian:") == ["q.ian"]


def test_parse_collects_placeholders(config: Config) -> None:
    loaded = parse(config, {"seeds": [], "seed_role": "ian: role", "h": {"ian": "x"}})
    assert loaded.placeholders == ("seed_role", "h.ian")


def test_parse_warns_about_unfilled_placeholders(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="qcal"):
        parse(config, {"seed_role": "ian: role"})
    assert "1 unfilled placeholder(s)" in caplog.text


def test_parse_uses_configured_placeholder_marker(make_config: Callable[[str], Config]) -> None:
    config = make_config('[experiments]\nplaceholder_marker = "TBD:"\n')
    assert parse(config, {"a": "TBD: x", "b": "ian: y"}).placeholders == ("a",)


def test_packaged_template_lists_every_open_decision(config: Config, repo: Path) -> None:
    text = (
        resources.files("qcal.resources").joinpath("templates/EXPERIMENTS.yaml").read_text("utf-8")
    )
    write(repo, "EXPERIMENTS.yaml", text)
    loaded = load_experiments(config)
    assert loaded.placeholders == (
        "question.ian",
        "hypotheses.H1.ian",
        "hypotheses.H2.ian",
        "hypotheses.H3.ian",
        "falsifier.ian",
        "statistics.ian",
        "splits.ian",
        "seed_role",
    )
    assert (loaded.cells, loaded.seeds) == ((), ())


def test_experiments_yaml_helper_round_trips(config: Config, repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml(CELLS, seeds=[4], extra_key="x"))
    loaded = load_experiments(config)
    assert loaded.seeds == (4,)
    assert yaml.safe_load(loaded.path.read_text("utf-8"))["extra_key"] == "x"
