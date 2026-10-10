"""Loading and validating the pre-registration (``EXPERIMENTS.yaml``).

The file is Ian-only. This module only reads it; it never writes it.
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qcal.config import Config
from qcal.log import get_logger

_log = get_logger("registry.experiments")
SCALAR_TYPES = (str, int, float, bool, type(None))  # what a cell factor may hold
_SCALARS = SCALAR_TYPES


class ExperimentsError(ValueError):
    """The pre-registration is missing or malformed."""


@dataclass(frozen=True)
class Cell:
    id: str
    factors: Mapping[str, Any]
    seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Experiments:
    path: Path
    data: Mapping[str, Any]
    cells: tuple[Cell, ...]
    seeds: tuple[int, ...]
    seed_role: str
    sha256: str
    placeholders: tuple[str, ...]

    def cell(self, cell_id: str) -> Cell:
        for cell in self.cells:
            if cell.id == cell_id:
                return cell
        raise ExperimentsError(f"cell {cell_id!r} is not pre-registered in {self.path.name}")

    def seeds_for(self, cell: Cell) -> tuple[int, ...]:
        return cell.seeds if cell.seeds is not None else self.seeds

    def match(self, patterns: str | Sequence[str]) -> list[Cell]:
        """Cells whose id matches any comma-separated or listed fnmatch pattern."""
        items = (
            [p.strip() for p in patterns.split(",")]
            if isinstance(patterns, str)
            else list(patterns)
        )
        items = [p for p in items if p]
        return [c for c in self.cells if any(fnmatch.fnmatchcase(c.id, p) for p in items)]


def load_experiments(config: Config, path: Path | None = None) -> Experiments:
    target = path or config.path("experiments")
    if not target.is_file():
        raise ExperimentsError(
            f"{target} does not exist. It is Ian's pre-registration; create it with `qcal init` "
            "and fill it in by hand."
        )
    raw = target.read_bytes()
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ExperimentsError(f"cannot parse {target}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ExperimentsError(f"{target} must contain a mapping at the top level")
    return parse_experiments(config, data, path=target, sha256=hashlib.sha256(raw).hexdigest())


def parse_experiments(
    config: Config, data: Mapping[str, Any], *, path: Path, sha256: str = ""
) -> Experiments:
    seeds = _seed_tuple(data.get(config.str_value("experiments.seeds_key"), []), "seeds")
    role = data.get(config.str_value("experiments.seed_role_key")) or config.str_value(
        "experiments.default_seed_role"
    )
    if not isinstance(role, str):
        raise ExperimentsError(f"seed_role must be a string, got {role!r}")
    cells = _parse_cells(config, data.get(config.str_value("experiments.cells_key")) or [])
    placeholders = tuple(
        find_placeholders(data, config.str_value("experiments.placeholder_marker"))
    )
    if placeholders:
        _log.warning("%s still has %d unfilled placeholder(s)", path.name, len(placeholders))
    return Experiments(path, data, cells, seeds, role, sha256, placeholders)


def _parse_cells(config: Config, raw: Any) -> tuple[Cell, ...]:
    if not isinstance(raw, list):
        raise ExperimentsError("cells must be a list")
    id_key = config.str_value("experiments.cell_id_key")
    seeds_key = config.str_value("experiments.seeds_key")
    cells: list[Cell] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ExperimentsError(f"cells[{index}] must be a mapping")
        cell_id = entry.get(id_key)
        if not isinstance(cell_id, str) or not cell_id:
            raise ExperimentsError(f"cells[{index}] needs a string {id_key!r}")
        if cell_id in seen:
            raise ExperimentsError(f"duplicate cell id {cell_id!r}")
        seen.add(cell_id)
        factors = {k: v for k, v in entry.items() if k not in {id_key, seeds_key}}
        bad = [k for k, v in factors.items() if not isinstance(v, _SCALARS)]
        if bad:
            raise ExperimentsError(f"cell {cell_id!r}: factors must be scalars ({', '.join(bad)})")
        seeds = (
            _seed_tuple(entry[seeds_key], f"cell {cell_id} seeds") if seeds_key in entry else None
        )
        cells.append(Cell(cell_id, factors, seeds))
    return tuple(cells)


def _seed_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(s, bool) or not isinstance(s, int) for s in value
    ):
        raise ExperimentsError(f"{label} must be a list of integers")
    if len(set(value)) != len(value):
        raise ExperimentsError(f"{label} contains duplicates")
    return tuple(value)


def find_placeholders(data: Any, marker: str, prefix: str = "") -> list[str]:
    """Dotted keys still holding Ian's placeholders.

    A placeholder is a key named after the marker (``ian``) or a string value that
    starts with it (``"ian: ..."``).
    """
    key_name = marker.rstrip(":")
    found: list[str] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if str(key) == key_name:
                found.append(dotted)  # the whole value is the placeholder; do not descend
                continue
            found.extend(find_placeholders(value, marker, dotted))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            found.extend(find_placeholders(value, marker, f"{prefix}[{i}]"))
    elif isinstance(data, str) and data.strip().startswith(marker):
        found.append(prefix)
    return found
