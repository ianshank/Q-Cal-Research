"""Design expansion: axes, exclude/fix/drop rules, content-addressed cell ids, summaries."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qcal.config import Config
from qcal.registry.cells import cell_id, cells_to_yaml_entries, expand_design, summarize
from qcal.registry.experiments import Cell, ExperimentsError, parse_experiments

PREFIX = "C-"
LENGTH = 10


def expand(design: dict[str, Any]) -> list[Cell]:
    return expand_design(design, prefix=PREFIX, length=LENGTH)


def factor_sets(cells: list[Cell]) -> list[dict[str, Any]]:
    return sorted((dict(c.factors) for c in cells), key=lambda f: json.dumps(f, sort_keys=True))


scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False),
    st.text(max_size=12),
)
factor_dicts = st.dictionaries(st.text(min_size=1, max_size=8), scalars, max_size=6)


# -- cell ids -----------------------------------------------------------------------------


def test_cell_id_is_prefix_plus_truncated_sha256_of_canonical_json() -> None:
    digest = hashlib.sha256(b'{"a":1,"b":"x"}').hexdigest()
    assert cell_id({"b": "x", "a": 1}, prefix="C-", length=10) == f"C-{digest[:10]}"


@pytest.mark.parametrize(("prefix", "length"), [("C-", 10), ("", 6), ("cell_", 64), ("X", 1)])
def test_cell_id_honours_prefix_and_length(prefix: str, length: int) -> None:
    value = cell_id({"detector": "atss"}, prefix=prefix, length=length)
    assert re.fullmatch(rf"{re.escape(prefix)}[0-9a-f]{{{length}}}", value)


@settings(database=None)
@given(factors=factor_dicts, data=st.data())
def test_cell_id_is_independent_of_insertion_order(factors: dict[str, Any], data: Any) -> None:
    reordered_keys = data.draw(st.permutations(list(factors)))
    reordered = {k: factors[k] for k in reordered_keys}
    assert cell_id(reordered, prefix=PREFIX, length=LENGTH) == cell_id(
        factors, prefix=PREFIX, length=LENGTH
    )


@settings(database=None)
@given(factors=factor_dicts)
def test_cell_id_is_deterministic(factors: dict[str, Any]) -> None:
    first = cell_id(factors, prefix=PREFIX, length=LENGTH)
    assert cell_id(dict(factors), prefix=PREFIX, length=LENGTH) == first


@settings(database=None)
@given(first=factor_dicts, second=factor_dicts)
def test_distinct_factor_sets_get_distinct_ids(
    first: dict[str, Any], second: dict[str, Any]
) -> None:
    assume(first != second)
    assert cell_id(first, prefix=PREFIX, length=LENGTH) != cell_id(
        second, prefix=PREFIX, length=LENGTH
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [({"a": 1}, {"a": "1"}), ({"a": 1}, {"a": 1.0}), ({"a": None}, {"a": "None"}), ({"a": 1}, {})],
)
def test_cell_id_distinguishes_scalar_types(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert cell_id(left, prefix=PREFIX, length=LENGTH) != cell_id(
        right, prefix=PREFIX, length=LENGTH
    )


# -- expansion ----------------------------------------------------------------------------


def test_expansion_is_the_cartesian_product() -> None:
    cells = expand({"axes": {"d": ["atss", "detr"], "p": ["fp32", "int8", "fp16"]}})
    assert len(cells) == 6
    assert {(c.factors["d"], c.factors["p"]) for c in cells} == {
        (d, p) for d in ("atss", "detr") for p in ("fp32", "int8", "fp16")
    }


def test_expanded_cells_are_sorted_by_content_addressed_id() -> None:
    cells = expand({"axes": {"d": ["atss", "detr", "yolox"], "p": ["fp32", "int8"]}})
    assert [c.id for c in cells] == sorted(c.id for c in cells)
    assert all(c.id == cell_id(c.factors, prefix=PREFIX, length=LENGTH) for c in cells)


def test_expanded_cells_have_no_seed_override() -> None:
    assert all(c.seeds is None for c in expand({"axes": {"d": ["a", "b"]}}))


def test_expansion_is_independent_of_axis_order() -> None:
    forward = expand({"axes": {"d": ["a", "b"], "p": [1, 2]}})
    backward = expand({"axes": {"p": [2, 1], "d": ["b", "a"]}})
    assert [c.id for c in forward] == [c.id for c in backward]


def test_adding_an_axis_value_never_renames_existing_cells() -> None:
    before = expand({"axes": {"d": ["atss", "detr"], "p": ["fp32"]}})
    after = expand({"axes": {"d": ["atss", "detr", "yolox"], "p": ["fp32"]}})
    assert {c.id for c in before} < {c.id for c in after}


@pytest.mark.parametrize(
    ("design", "message"),
    [
        ({}, "design.axes must be a non-empty mapping"),
        ({"axes": {}}, "design.axes must be a non-empty mapping"),
        ({"axes": [["d", "a"]]}, "design.axes must be a non-empty mapping"),
        ({"axes": {"d": []}}, "design.axes.d must be a non-empty list"),
        ({"axes": {"d": "atss"}}, "design.axes.d must be a non-empty list"),
        ({"axes": {"d": ["a"]}, "rules": {"exclude": {"d": "a"}}}, "design.rules must be a list"),
        ({"axes": {"d": ["a"]}, "rules": ["exclude d=a"]}, "each design rule must be a mapping"),
        ({"axes": {"d": ["a"]}, "rules": [{"fix": {"x": 1}}]}, "needs 'exclude' or a 'when'"),
        ({"axes": {"d": ["a"]}, "rules": [{"when": "d=a", "fix": {}}]}, "needs 'exclude' or a"),
        ({"axes": {"d": ["a"]}, "rules": [{"when": {"d": "a"}}]}, "needs 'fix' or 'drop'"),
        ({"axes": {"d": ["a"]}, "rules": [{"exclude": "d=a"}]}, "'exclude' must be a mapping"),
        ({"axes": {"d": ["a"]}, "rules": [{"exclude": ["a"]}]}, "'exclude' must be a mapping"),
    ],
)
def test_invalid_designs_are_rejected(design: dict[str, Any], message: str) -> None:
    with pytest.raises(ExperimentsError, match=re.escape(message)):
        expand(design)


def test_when_rule_without_action_is_rejected_even_if_it_matches_nothing() -> None:
    with pytest.raises(ExperimentsError, match="needs 'fix' or 'drop'"):
        expand({"axes": {"d": ["a"]}, "rules": [{"when": {"d": "zzz"}}]})


def test_rules_are_validated_even_after_every_cell_is_excluded() -> None:
    with pytest.raises(ExperimentsError, match="each design rule must be a mapping"):
        expand({"axes": {"d": ["a"]}, "rules": [{"exclude": {"d": "a"}}, "not a rule"]})


@pytest.mark.parametrize("rules", [None, []])
def test_missing_rules_mean_the_full_product(rules: Any) -> None:
    assert len(expand({"axes": {"d": ["a", "b"], "p": [1, 2]}, "rules": rules})) == 4


def test_exclude_rule_drops_matching_cells() -> None:
    cells = expand(
        {"axes": {"d": ["a", "b"], "p": [1, 2]}, "rules": [{"exclude": {"d": "b", "p": 2}}]}
    )
    assert factor_sets(cells) == [{"d": "a", "p": 1}, {"d": "a", "p": 2}, {"d": "b", "p": 1}]


def test_exclude_rule_with_list_matches_membership() -> None:
    cells = expand({"axes": {"d": ["a", "b", "c"]}, "rules": [{"exclude": {"d": ["a", "c"]}}]})
    assert factor_sets(cells) == [{"d": "b"}]


def test_exclude_rule_on_unknown_factor_matches_nothing() -> None:
    cells = expand({"axes": {"d": ["a", "b"]}, "rules": [{"exclude": {"missing": "a"}}]})
    assert len(cells) == 2


def test_excluding_everything_gives_no_cells() -> None:
    assert expand({"axes": {"d": ["a"]}, "rules": [{"exclude": {"d": "a"}}]}) == []


def test_fix_rule_sets_factors_on_matching_cells() -> None:
    cells = expand(
        {
            "axes": {"precision": ["fp32", "int8"], "fit": ["fp32", "int8"]},
            "rules": [{"when": {"precision": "fp32"}, "fix": {"fit": "fp32"}}],
        }
    )
    assert factor_sets(cells) == [
        {"fit": "fp32", "precision": "fp32"},
        {"fit": "fp32", "precision": "int8"},
        {"fit": "int8", "precision": "int8"},
    ]


def test_fix_rule_can_add_a_new_factor() -> None:
    cells = expand({"axes": {"d": ["a"]}, "rules": [{"when": {"d": "a"}, "fix": {"tf32": False}}]})
    assert factor_sets(cells) == [{"d": "a", "tf32": False}]


def test_drop_rule_removes_irrelevant_factors_and_deduplicates() -> None:
    cells = expand(
        {
            "axes": {"target": ["torch", "trt"], "tf32": [True, False]},
            "rules": [{"when": {"target": "trt"}, "drop": ["tf32"]}],
        }
    )
    assert factor_sets(cells) == [
        {"target": "torch", "tf32": False},
        {"target": "torch", "tf32": True},
        {"target": "trt"},
    ]


def test_drop_of_absent_factor_is_ignored() -> None:
    cells = expand({"axes": {"d": ["a"]}, "rules": [{"when": {"d": "a"}, "drop": ["nope"]}]})
    assert factor_sets(cells) == [{"d": "a"}]


def test_when_rule_with_list_matches_membership() -> None:
    cells = expand(
        {
            "axes": {"d": ["a", "b", "c"], "x": [1]},
            "rules": [{"when": {"d": ["a", "b"]}, "fix": {"x": 0}}],
        }
    )
    assert factor_sets(cells) == [{"d": "a", "x": 0}, {"d": "b", "x": 0}, {"d": "c", "x": 1}]


def test_rule_can_fix_and_drop_at_once() -> None:
    cells = expand(
        {
            "axes": {"d": ["a"], "y": [1, 2], "z": [3]},
            "rules": [{"when": {"d": "a"}, "fix": {"z": 9}, "drop": ["y"]}],
        }
    )
    assert factor_sets(cells) == [{"d": "a", "z": 9}]


def test_rules_apply_in_order() -> None:
    cells = expand(
        {
            "axes": {"d": ["a", "b"], "y": [1]},
            "rules": [
                {"when": {"d": "a"}, "drop": ["y"]},
                {"exclude": {"y": 1}},
            ],
        }
    )
    assert factor_sets(cells) == [{"d": "a"}]


def test_rules_after_an_exclusion_are_not_applied() -> None:
    cells = expand(
        {
            "axes": {"d": ["a", "b"]},
            "rules": [{"exclude": {"d": "a"}}, {"when": {"d": "a"}, "fix": {"d": "b2"}}],
        }
    )
    assert factor_sets(cells) == [{"d": "b"}]


# -- summaries and emission ---------------------------------------------------------------


def test_summarize_counts_values_per_factor_sorted() -> None:
    cells = [
        Cell("C-1", {"p": "int8", "d": "b"}),
        Cell("C-2", {"p": "fp32", "d": "a"}),
        Cell("C-3", {"p": "int8"}),
    ]
    summary = summarize(cells)
    assert summary == {"d": {"a": 1, "b": 1}, "p": {"fp32": 1, "int8": 2}}
    assert list(summary) == ["d", "p"]
    assert list(summary["p"]) == ["fp32", "int8"]


def test_summarize_stringifies_values() -> None:
    cells = [Cell("C-1", {"tf32": True, "n": 500}), Cell("C-2", {"tf32": None, "n": 500})]
    assert summarize(cells) == {"n": {"500": 2}, "tf32": {"None": 1, "True": 1}}


def test_summarize_of_no_cells_is_empty() -> None:
    assert summarize([]) == {}


def test_cells_to_yaml_entries_put_the_id_first() -> None:
    entries = cells_to_yaml_entries([Cell("C-1", {"d": "a", "p": 1})], "id")
    assert entries == [{"id": "C-1", "d": "a", "p": 1}]
    assert next(iter(entries[0])) == "id"


def test_emitted_entries_parse_back_to_the_same_cells(config: Config) -> None:
    cells = expand(
        {
            "axes": {"d": ["a", "b"], "tf32": [True, False], "n": [500, 2000]},
            "rules": [{"when": {"d": "b"}, "drop": ["tf32"]}],
        }
    )
    entries = cells_to_yaml_entries(cells, config.str_value("experiments.cell_id_key"))
    parsed = parse_experiments(
        config, {"seeds": [0], "cells": entries}, path=config.path("experiments")
    )
    assert parsed.cells == tuple(cells)
