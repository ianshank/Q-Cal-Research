"""Expand a design (axes + rules) into an explicit, frozen cell list.

The Review's axes form a 16,200-cell Cartesian product (plan §10.2). This
expander makes the product explicit, applies pre-registered pruning rules, and
assigns content-addressed cell ids so adding an axis value never renames
existing cells.

Rule forms (all keys are factor names; values may be scalars or lists):
  {exclude: {k: v}}                    drop cells matching every key
  {when: {k: v}, fix: {k2: v2}}         set factors on matching cells
  {when: {k: v}, drop: [k2, ...]}       remove factors irrelevant to matching cells
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from qcal.registry.experiments import Cell, ExperimentsError


def cell_id(factors: Mapping[str, Any], *, prefix: str, length: int) -> str:
    canonical = json.dumps(dict(factors), sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}{hashlib.sha256(canonical.encode()).hexdigest()[:length]}"


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else [value]


def _matches(factors: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    return all(k in factors and factors[k] in _as_list(v) for k, v in condition.items())


def expand_design(design: Mapping[str, Any], *, prefix: str, length: int) -> list[Cell]:
    axes = design.get("axes")
    if not isinstance(axes, Mapping) or not axes:
        raise ExperimentsError("design.axes must be a non-empty mapping of factor -> values")
    for name, values in axes.items():
        if not isinstance(values, list) or not values:
            raise ExperimentsError(f"design.axes.{name} must be a non-empty list")
    rules = design.get("rules", []) or []
    if not isinstance(rules, list):
        raise ExperimentsError("design.rules must be a list")
    for rule in rules:
        _validate_rule(rule)

    names = list(axes)
    seen: dict[str, Cell] = {}
    for combo in itertools.product(*(axes[n] for n in names)):
        factors: dict[str, Any] | None = dict(zip(names, combo, strict=True))
        for rule in rules:
            factors = _apply(rule, factors) if factors is not None else None
        if factors is None:
            continue
        cid = cell_id(factors, prefix=prefix, length=length)
        seen.setdefault(cid, Cell(cid, factors))
    return sorted(seen.values(), key=lambda c: c.id)


def _validate_rule(rule: Any) -> None:
    if not isinstance(rule, Mapping):
        raise ExperimentsError("each design rule must be a mapping")
    if "exclude" in rule:
        if not isinstance(rule["exclude"], Mapping):
            raise ExperimentsError(f"rule {dict(rule)!r}: 'exclude' must be a mapping")
        return
    if not isinstance(rule.get("when"), Mapping):
        raise ExperimentsError(f"rule {dict(rule)!r} needs 'exclude' or a 'when' mapping")
    if "fix" not in rule and "drop" not in rule:
        raise ExperimentsError(f"rule {dict(rule)!r} needs 'fix' or 'drop'")
    if "fix" in rule and not isinstance(rule["fix"], Mapping):
        raise ExperimentsError(f"rule {dict(rule)!r}: 'fix' must be a mapping")
    if not isinstance(rule.get("drop", []) or [], list):
        raise ExperimentsError(f"rule {dict(rule)!r}: 'drop' must be a list")


def _apply(rule: Mapping[str, Any], factors: dict[str, Any]) -> dict[str, Any] | None:
    if "exclude" in rule:
        return None if _matches(factors, rule["exclude"]) else factors
    if not _matches(factors, rule["when"]):
        return factors
    updated = dict(factors)
    updated.update(rule.get("fix", {}) or {})
    for key in rule.get("drop", []) or []:
        updated.pop(key, None)
    return updated


def summarize(cells: Sequence[Cell]) -> dict[str, dict[str, int]]:
    summary: dict[str, Counter[str]] = {}
    for cell in cells:
        for key, value in cell.factors.items():
            summary.setdefault(key, Counter())[str(value)] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(summary.items())}


def cells_to_yaml_entries(cells: Sequence[Cell], id_key: str) -> list[dict[str, Any]]:
    return [{id_key: c.id, **dict(c.factors)} for c in cells]
