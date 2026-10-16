"""Named aggregations shared by the table builder and the claims checker.

Both sides compute a cell value through the same function, so a table value and
its verification can never disagree by construction.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from qcal.components import ComponentRegistry
from qcal.config import Config, ConfigError
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


#: How an identity column is named in a refusal; others are named by the column itself.
_IDENTITY_NAMES: Final = {CONFIG_INPUTS_COLUMN: "configuration files"}


def identity_column_problems(
    columns: Sequence[str], core: Sequence[str], prefixes: Sequence[str]
) -> list[str]:
    """Identity columns that no index can hold: neither core nor under a column prefix."""
    return [
        f"registry.identity_columns: {c!r} is neither in registry.core_columns nor under a "
        f"column prefix {sorted(prefixes)}"
        for c in columns
        if c not in core and not any(c.startswith(p) for p in prefixes)
    ]


def identity_columns(config: Config) -> tuple[str, ...]:
    """``registry.identity_columns``, validated against the index's columns."""
    columns = config.str_list("registry.identity_columns")
    prefixes = [str(v) for v in config.section("registry.column_prefixes").values()]
    problems = identity_column_problems(columns, config.str_list("registry.core_columns"), prefixes)
    if problems:
        raise ConfigError("; ".join(problems))
    return tuple(columns)


def nullable_column_problems(nullable: Sequence[str], columns: Sequence[str]) -> list[str]:
    """``registry.identity_nullable`` entries that are not identity columns."""
    return [
        f"registry.identity_nullable: {c!r} is not in registry.identity_columns"
        for c in nullable
        if c not in columns
    ]


def identity_nullable(config: Config) -> tuple[str, ...]:
    """Identity columns a run may leave empty (``registry.identity_nullable``), validated."""
    nullable = config.str_list("registry.identity_nullable")
    problems = nullable_column_problems(nullable, identity_columns(config))
    if problems:
        raise ConfigError("; ".join(problems))
    return tuple(nullable)


def identity_problem(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    *,
    core: Sequence[str] = (),
    nullable: Sequence[str] = (),
) -> str | None:
    """Why ``rows`` are not runs of one experiment, or ``None``.

    Fails closed: an identity column missing from every row is a problem, not a match. For a
    core column the index is stale; for a prefixed one (``env.*``) no run recorded it, so it is
    misspelt in ``registry.identity_columns`` or the program stopped writing it. An empty
    value (a run that did not record the column; the index holds ``""`` for it) matches
    nothing either, unless the column is in ``registry.identity_nullable``. One run is
    trivially one experiment.
    """
    if len(rows) < 2:
        return None
    ids = ", ".join(sorted(r["run_id"] for r in rows))
    for column in columns:
        if all(column not in row for row in rows):
            if column in core:
                return f"the index has no {column} column; regenerate it (qcal registry index)"
            return (
                f"no run in {ids} recorded {column}: check registry.identity_columns for a "
                "misspelling, or whether the program still writes it"
            )
        empty = sorted(r["run_id"] for r in rows if row_value(r, column) == "")
        if empty and column not in nullable:
            return (
                f"runs {', '.join(empty)} recorded no {column}, so they cannot be shown to be "
                "the same experiment (an empty value matches nothing; registry.identity_nullable "
                "lists the columns a run may leave empty)"
            )
        values = sorted({row_value(row, column) for row in rows})
        if len(values) > 1:
            shown = ", ".join(v[:12] or "(none)" for v in values)
            what = _IDENTITY_NAMES.get(column, f"{column} values")
            return f"runs {ids} ran under different {what} ({column}: {shown})"
    return None


def row_value(row: Mapping[str, str], column: str) -> str:
    """An index cell as text: absent and ``None`` (a short CSV row) are both empty."""
    value = row.get(column)
    return "" if value is None else str(value)


def unsound_aggregate(
    name: str,
    rows: Sequence[Mapping[str, str]],
    *,
    environment_prefix: str,
    identity: Sequence[str] = (CONFIG_INPUTS_COLUMN,),
    core: Sequence[str] = (CONFIG_INPUTS_COLUMN,),
    nullable: Sequence[str] = (),
) -> str | None:
    """Why aggregating these index rows would give a wrong number, or ``None``.

    Runs that differ in an identity column (``registry.identity_columns``) are different
    experiments, so they are not averaged. A spread over seeds that changed nothing is a
    false zero variance.
    """
    problem = identity_problem(rows, identity, core=core, nullable=nullable)
    if problem:
        return f"{problem}; rerun the stale ones"
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
