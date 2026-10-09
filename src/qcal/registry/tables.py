"""LaTeX result tables generated only from the run index.

Every number is emitted as ``\\<macro>{<ref>}{<value>}`` where ``<ref>`` is
``agg:<fn>:<metric>:<run>+<run>...`` (or ``run:<id>:<metric>`` for one run), so
the claims checker can recompute it. Table layouts live in TOML spec files under
``paths.table_specs_dir``; no numbers or layouts are hard-coded here.

Stdlib-only.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.config import Config, ConfigError
from qcal.integrity.aggregates import aggregate, format_value
from qcal.log import get_logger
from qcal.registry.index import read_index, render_value

_log = get_logger("registry.tables")
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
}


@dataclass(frozen=True)
class ColumnSpec:
    metric: str
    agg: str
    digits: int
    header: str


@dataclass(frozen=True)
class TableSpec:
    name: str
    rows: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    row_headers: tuple[str, ...] = ()
    caption: str = ""
    label: str = ""
    filters: Mapping[str, Any] = field(default_factory=dict)
    statuses: tuple[str, ...] = ()
    alignment: str = ""


@dataclass(frozen=True)
class TableResult:
    path: Path
    changed: bool


def latex_escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


def make_ref(agg: str, metric: str, run_ids: Sequence[str]) -> str:
    if len(run_ids) == 1 and agg in {"mean", "median", "min", "max", "sum"}:
        return f"run:{run_ids[0]}:{metric}"
    return f"agg:{agg}:{metric}:{'+'.join(run_ids)}"


def load_specs(config: Config) -> list[TableSpec]:
    directory = config.path("table_specs_dir")
    if not directory.is_dir():
        return []
    specs: list[TableSpec] = []
    for path in sorted(directory.glob(config.str_value("tables.spec_glob"))):
        data = tomllib.loads(path.read_text("utf-8"))
        entries = data.get("table", [data])
        if not isinstance(entries, list):
            raise ConfigError(f"{path}: 'table' must be an array of tables")
        specs.extend(_parse_spec(config, entry, path) for entry in entries)
    names = [s.name for s in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"duplicate table names: {', '.join(duplicates)}")
    return specs


def _parse_spec(config: Config, data: Mapping[str, Any], origin: Path) -> TableSpec:
    try:
        name = str(data["name"])
        rows = tuple(_column_name(config, "factors", r) for r in data["rows"])
        raw_columns = data["columns"]
    except KeyError as exc:
        raise ConfigError(f"{origin}: table spec is missing {exc}") from None
    columns = tuple(
        ColumnSpec(
            metric=str(c["metric"]),
            agg=str(c.get("agg", config.str_value("tables.default_agg"))),
            digits=int(c.get("digits", config.int_value("tables.default_digits"))),
            header=str(c.get("header", c["metric"])),
        )
        for c in raw_columns
    )
    if not rows or not columns:
        raise ConfigError(f"{origin}: table {name!r} needs at least one row key and one column")
    return TableSpec(
        name=name,
        rows=rows,
        columns=columns,
        row_headers=tuple(str(h) for h in data.get("row_headers", [])),
        caption=str(data.get("caption", "")),
        label=str(data.get("label", "")),
        filters={_column_name(config, "factors", k): v for k, v in data.get("filter", {}).items()},
        statuses=tuple(data.get("status", [config.str_value("registry.ok_status")])),
        alignment=str(data.get("alignment", "")),
    )


def _column_name(config: Config, group: str, name: str) -> str:
    prefix = str(config.get(f"registry.column_prefixes.{group}"))
    known = [str(v) for v in config.section("registry.column_prefixes").values()]
    return name if any(name.startswith(p) for p in known) else f"{prefix}{name}"


def _selected(row: Mapping[str, str], spec: TableSpec, superseded: set[str]) -> bool:
    if row["run_id"] in superseded or row.get("status") not in spec.statuses:
        return False
    for key, wanted in spec.filters.items():
        options = [render_value(v) for v in (wanted if isinstance(wanted, list) else [wanted])]
        if row.get(key, "") not in options:
            return False
    return True


def render_table(config: Config, spec: TableSpec, index: Sequence[Mapping[str, str]]) -> str:
    macro = config.str_value("tables.macro")
    metric_prefix = str(config.get("registry.column_prefixes.metrics"))
    missing = config.str_value("tables.missing_cell")
    superseded = {r["supersedes"] for r in index if r.get("supersedes")}
    groups: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in index:
        if _selected(row, spec, superseded):
            groups.setdefault(tuple(row.get(k, "") for k in spec.rows), []).append(row)

    headers = list(spec.row_headers) or [r.split(".", 1)[-1] for r in spec.rows]
    align = spec.alignment or "l" * len(spec.rows) + "r" * len(spec.columns)
    rule = config.str_value("tables.rule")
    lines = [
        "% Generated by `qcal registry tables` from the run index. Do not edit by hand.",
        f"\\providecommand{{\\{macro}}}[2]{{#2}}",
        f"\\begin{{table}}[{config.str_value('tables.float_placement')}]",
        "\\centering",
    ]
    if spec.caption:
        lines.append(f"\\caption{{{spec.caption}}}")
    if spec.label:
        lines.append(f"\\label{{{spec.label}}}")
    lines += [
        f"\\begin{{tabular}}{{{align}}}",
        rule,
        " & ".join(headers + [c.header for c in spec.columns]) + r" \\",
        rule,
    ]
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: r["run_id"])
        cells = [latex_escape(k) for k in key]
        for column in spec.columns:
            column_key = f"{metric_prefix}{column.metric}"
            usable = [r for r in rows if r.get(column_key, "") != ""]
            if not usable or (column.agg == "std" and len(usable) < 2):
                cells.append(missing)
                continue
            value = aggregate(column.agg, [float(r[column_key]) for r in usable])
            ref = make_ref(column.agg, column.metric, [r["run_id"] for r in usable])
            cells.append(f"\\{macro}{{{ref}}}{{{format_value(value, column.digits)}}}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [rule, "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def build_tables(config: Config, *, check: bool = False) -> list[TableResult]:
    index = read_index(config.path("index_csv"))
    out_dir = config.path("tables_dir")
    results: list[TableResult] = []
    for spec in load_specs(config):
        target = out_dir / f"{spec.name}.tex"
        text = render_table(config, spec, index)
        changed = not target.is_file() or target.read_text("utf-8") != text
        if changed and not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            _log.info("wrote %s", target)
        elif changed:
            _log.error("%s is stale; run `qcal registry tables`", target)
        results.append(TableResult(target, changed))
    return results
