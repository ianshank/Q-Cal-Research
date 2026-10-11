"""The derived run index: a deterministic flattening of every run record.

``runs/index.csv`` is regenerated from ``runs/registry/*.json`` and committed.
Records are immutable (CI enforces it); the index is checked for consistency by
regenerating it, so adding a metric column never counts as tampering.

Stdlib-only apart from the optional Parquet export.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qcal.config import Config, ConfigError
from qcal.log import get_logger
from qcal.registry.records import RunRecord
from qcal.registry.store import RegistryStore

_log = get_logger("registry.index")
_PREFIX_GROUPS = ("factors", "metrics", "environment", "resources")


@dataclass(frozen=True)
class IndexResult:
    path: Path
    rows: int
    changed: bool
    parquet: Path | None = None


def render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def columns_for(
    rows: Sequence[Mapping[str, Any]], core: Sequence[str], prefixes: Mapping[str, str]
) -> list[str]:
    present = {key for row in rows for key in row}
    ordered = list(core)
    rank = {prefixes.get(g, f"{g}."): i for i, g in enumerate(_PREFIX_GROUPS)}

    def sort_key(name: str) -> tuple[int, str]:
        for prefix, position in rank.items():
            if name.startswith(prefix):
                return position, name
        return len(rank), name

    ordered.extend(sorted((c for c in present if c not in set(core)), key=sort_key))
    return ordered


def render_csv(records: Iterable[RunRecord], config: Config) -> tuple[str, int]:
    prefixes = {k: str(v) for k, v in config.section("registry.column_prefixes").items()}
    rows = [r.flat(prefixes) for r in sorted(records, key=lambda r: r.run_id)]
    columns = columns_for(rows, config.str_list("registry.core_columns"), prefixes)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([render_value(row.get(c)) for c in columns])
    return buffer.getvalue(), len(rows)


def write_index(config: Config, store: RegistryStore, *, check: bool = False) -> IndexResult:
    """Regenerate the index. With ``check`` only compare and report ``changed``."""
    target = config.path("index_csv")
    records = store.load_all(strict=True)
    text, count = render_csv(records, config)
    current = target.read_text("utf-8") if target.is_file() else None
    # No records and no index yet is consistent: there is nothing to index.
    changed = current != text and not (count == 0 and current is None)
    if check:
        if changed:
            _log.error("%s is stale; run `qcal registry index`", target)
        return IndexResult(target, count, changed)
    if changed:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        _log.info("wrote %s (%d rows)", target, count)
    parquet = _maybe_parquet(config, text, csv_changed=changed) if count else None
    return IndexResult(target, count, changed, parquet)


def superseded_ids(rows: Iterable[Mapping[str, str]]) -> frozenset[str]:
    """Run ids that a later row supersedes (the index view of ``records.effective``)."""
    return frozenset(r["supersedes"] for r in rows if r.get("supersedes"))


def read_index(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _maybe_parquet(config: Config, csv_text: str, *, csv_changed: bool) -> Path | None:
    mode = config.str_value("registry.write_parquet")
    if mode not in {"auto", "always", "never"}:
        raise ConfigError("registry.write_parquet must be auto, always or never")
    if mode == "never":
        return None
    if importlib.util.find_spec("pyarrow") is None:
        if mode == "always":
            raise ConfigError("registry.write_parquet=always but pyarrow is not installed")
        _log.debug("pyarrow not installed; skipping Parquet export")
        return None
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    target = config.path("index_parquet")
    csv_path = config.path("index_csv")
    stale = not target.is_file() or (
        csv_path.is_file() and target.stat().st_mtime < csv_path.stat().st_mtime
    )
    if not csv_changed and not stale:
        _log.debug("%s is current", target)
        return target
    table = pacsv.read_csv(io.BytesIO(csv_text.encode("utf-8")))
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)
    _log.info("wrote %s", target)
    return target
