"""Named aggregations and the single number-formatting rule shared with the claims checker."""

from __future__ import annotations

import statistics

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qcal.components import UnknownComponentError
from qcal.integrity.aggregates import AGGREGATES, aggregate, format_value

VALUES = [4.0, 1.0, 3.0, 2.0]


def test_registry_exposes_the_documented_aggregations() -> None:
    assert AGGREGATES.names() == ["count", "max", "mean", "median", "min", "std", "sum"]


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        ("mean", VALUES, 2.5),
        ("median", [3.0, 1.0, 2.0], 2.0),
        ("median", VALUES, 2.5),
        ("min", VALUES, 1.0),
        ("max", VALUES, 4.0),
        ("sum", VALUES, 10.0),
        ("count", VALUES, 4.0),
        (
            "std",
            [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
            statistics.stdev([2, 4, 4, 4, 5, 5, 7, 9]),
        ),
    ],
)
def test_aggregate_computes_named_function(name: str, values: list[float], expected: float) -> None:
    assert aggregate(name, values) == pytest.approx(expected)


def test_std_is_the_sample_standard_deviation() -> None:
    assert aggregate("std", [1.0, 3.0]) == pytest.approx(2**0.5)


@pytest.mark.parametrize("name", ["min", "max", "sum", "count", "median"])
def test_aggregate_always_returns_a_float(name: str) -> None:
    assert type(aggregate(name, [3, 1, 2])) is float


@pytest.mark.parametrize("name", ["mean", "median", "min", "max", "std", "sum", "count"])
def test_aggregate_over_no_values_is_an_error(name: str) -> None:
    with pytest.raises(ValueError, match=f"cannot aggregate {name} over no values"):
        aggregate(name, [])


def test_std_of_a_single_value_is_an_error() -> None:
    with pytest.raises(ValueError, match="std needs at least two values"):
        aggregate("std", [1.0])


def test_unknown_aggregation_is_rejected() -> None:
    with pytest.raises(UnknownComponentError, match="unknown aggregate 'mode'"):
        aggregate("mode", [1.0])


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (12.345678, 2, "12.35"),
        (3, 3, "3.000"),
        (-0.04, 1, "-0.0"),
        (2.5, 0, "2"),
        (0.125, 2, "0.12"),
        (41.0, 1, "41.0"),
    ],
)
def test_format_value_uses_fixed_point_with_given_digits(
    value: float, digits: int, expected: str
) -> None:
    assert format_value(value, digits) == expected


@settings(database=None)
@given(
    value=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False),
    digits=st.integers(min_value=0, max_value=6),
)
def test_format_value_is_within_half_a_unit_of_the_last_digit(value: float, digits: int) -> None:
    rendered = format_value(value, digits)
    assert abs(float(rendered) - value) <= 0.5 * 10**-digits + 1e-9 * max(1.0, abs(value))
