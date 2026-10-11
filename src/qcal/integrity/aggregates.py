"""Named aggregations shared by the table builder and the claims checker.

Both sides compute a cell value through the same function, so a table value and
its verification can never disagree by construction.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from qcal.components import ComponentRegistry
from qcal.registry.records import CONFIG_INPUTS_COLUMN, SEED_EFFECTIVE_KEY

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


#: Aggregates that measure spread across seeds; meaningless over identical runs.
SPREAD_AGGREGATES: Final = frozenset({"std"})


def unsound_aggregate(
    name: str, rows: Sequence[Mapping[str, str]], *, environment_prefix: str
) -> str | None:
    """Why aggregating these index rows would give a wrong number, or ``None``.

    Runs under different configuration files are different experiments, so they are not
    averaged. A spread over seeds that changed nothing is a false zero variance.
    """
    digests = sorted({row.get(CONFIG_INPUTS_COLUMN, "") for row in rows})
    if len(digests) > 1:
        shown = ", ".join(d[:12] or "(none)" for d in digests)
        return (
            f"runs {', '.join(sorted(r['run_id'] for r in rows))} ran under different "
            f"configuration files ({CONFIG_INPUTS_COLUMN}: {shown}); rerun the stale ones"
        )
    column = f"{environment_prefix}{SEED_EFFECTIVE_KEY}"
    if name in SPREAD_AGGREGATES and any(row.get(column) == "false" for row in rows):
        return (
            f"{name} over runs whose seed changed nothing ({column} is false) would report a "
            "false zero variance"
        )
    return None


def format_value(value: float, digits: int) -> str:
    """The single formatting rule used when writing and when verifying numbers."""
    return f"{value:.{digits}f}"
