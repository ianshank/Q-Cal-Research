"""Named aggregations shared by the table builder and the claims checker.

Both sides compute a cell value through the same function, so a table value and
its verification can never disagree by construction.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence

from qcal.components import ComponentRegistry

Aggregate = Callable[[Sequence[float]], float]
AGGREGATES: ComponentRegistry[Aggregate] = ComponentRegistry("aggregate")

AGGREGATES.register("mean", statistics.fmean)
AGGREGATES.register("median", statistics.median)
AGGREGATES.register("min", min)
AGGREGATES.register("max", max)
AGGREGATES.register("std", statistics.stdev)
AGGREGATES.register("sum", sum)
AGGREGATES.register("count", lambda values: float(len(values)))


def aggregate(name: str, values: Sequence[float]) -> float:
    if not values:
        raise ValueError(f"cannot aggregate {name} over no values")
    if name == "std" and len(values) < 2:
        raise ValueError("std needs at least two values")
    return float(AGGREGATES.get(name)(values))


def format_value(value: float, digits: int) -> str:
    """The single formatting rule used when writing and when verifying numbers."""
    return f"{value:.{digits}f}"
