"""Verify that every result number in the paper and claim documents traces to the registry.

Two kinds of finding:

* a tagged value whose reference does not resolve, references a superseded run,
  or does not equal the registry value at the displayed precision;
* an untagged decimal in a *strict* file (paper sections and tables, CLAIMS.md),
  after removing configured layout, citation and version contexts.

Whole numbers are flagged only when written as results (``7 points``, ``3\\%``).
The escape hatches, a line carrying the ignore marker (``qcal:ignore``) and
``\\qcalfixed{...}`` for declared constants, work only in files of the policy
categories named by ``claims.escape_hatch_categories`` (Ian-only by default), so an
agent cannot use them to hide a number. Tagged values are verified on every line.

Stdlib-only: imported by the Claude Code Stop hook.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qcal.config import Config
from qcal.globs import first_match, iter_files
from qcal.integrity.aggregates import aggregate, format_value, unsound_aggregate
from qcal.log import get_logger
from qcal.policy import Policy
from qcal.registry.index import read_index, superseded_ids

_log = get_logger("integrity.claims")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    kind: str
    message: str

    def render(self, root: Path) -> str:
        shown = self.path.relative_to(root) if self.path.is_relative_to(root) else self.path
        return f"{shown}:{self.line}: [{self.kind}] {self.message}"


@dataclass(frozen=True)
class _Context:
    rows: Mapping[str, Mapping[str, str]]
    superseded: frozenset[str]
    metric_prefix: str
    tolerance: float
    superseded_is_error: bool
    displayed: re.Pattern[str] = re.compile(r"\s*(?P<number>-?(?:\d+(?:\.\d+)?|\.\d+))\s*")
    # cell id -> run ids of its current ok runs; empty disables the completeness rule
    current_by_cell: Mapping[str, frozenset[str]] = field(default_factory=dict)
    allow_pooled_cells: bool = False
    environment_prefix: str = "env."


class _RefError(ValueError):
    pass


def _parse_ref(ref: str) -> tuple[str, str, list[str]]:
    parts = ref.split(":")
    if parts[0] == "run" and len(parts) == 3:
        return "mean", parts[2], [parts[1]]
    if parts[0] == "agg" and len(parts) == 4:
        run_ids = [r for r in parts[3].split("+") if r]
        repeated = sorted({r for r in run_ids if run_ids.count(r) > 1})
        if repeated:
            raise _RefError(f"{ref} names {', '.join(repeated)} more than once (weights a seed)")
        return parts[1], parts[2], run_ids
    raise _RefError(
        f"malformed reference {ref!r}; expected run:<id>:<metric> or agg:<fn>:<metric>:<ids>"
    )


def _values(ref: str, metric: str, run_ids: Sequence[str], ctx: _Context) -> list[float]:
    values: list[float] = []
    for run_id in run_ids:
        row = ctx.rows.get(run_id)
        if row is None:
            raise _RefError(f"{ref} names unknown run {run_id!r}")
        if run_id in ctx.superseded and ctx.superseded_is_error:
            raise _RefError(f"{ref} uses superseded run {run_id!r}")
        raw = row.get(f"{ctx.metric_prefix}{metric}", "")
        if raw == "":
            raise _RefError(f"run {run_id!r} has no metric {metric!r}")
        values.append(float(raw))
    return values


def _require_complete(ref: str, run_ids: Sequence[str], ctx: _Context) -> None:
    """A reference names every current run of its cell (no seed cherry-picking), and one
    cell unless pooling is allowed (no averaging across evaluation targets)."""
    if not ctx.current_by_cell:
        return
    cells = sorted({ctx.rows[r].get("cell_id", "") for r in run_ids})
    if len(cells) > 1 and not ctx.allow_pooled_cells:
        raise _RefError(f"{ref} pools runs of cells {', '.join(cells)}")
    expected = frozenset().union(*(ctx.current_by_cell.get(c, frozenset()) for c in cells))
    if frozenset(run_ids) != expected:
        missing = sorted(expected - set(run_ids))
        extra = sorted(set(run_ids) - expected)
        detail = f"leaves out {', '.join(missing)}" if missing else ""
        if extra:
            detail += ("; " if detail else "") + f"includes non-current {', '.join(extra)}"
        raise _RefError(
            f"{ref} must name every current run of {', '.join(cells)} ({detail}); "
            f"use agg:<fn>:<metric>:{'+'.join(sorted(expected))}"
        )


def current_runs_by_cell(
    index: Sequence[Mapping[str, str]], superseded: frozenset[str], ok_status: str
) -> dict[str, frozenset[str]]:
    cells: dict[str, set[str]] = {}
    for row in index:
        if row.get("status") == ok_status and row["run_id"] not in superseded:
            cells.setdefault(row.get("cell_id", ""), set()).add(row["run_id"])
    return {cell: frozenset(ids) for cell, ids in cells.items()}


def verify_reference(ref: str, displayed: str, ctx: _Context) -> str | None:
    """Return an error message, or ``None`` when ``displayed`` matches the registry."""
    number = ctx.displayed.fullmatch(displayed)
    if number is None:
        return f"value {displayed!r} for {ref} must be exactly one number"
    shown = number.group("number")
    digits = len(shown.split(".", 1)[1]) if "." in shown else 0
    try:
        agg, metric, run_ids = _parse_ref(ref)
        expected = aggregate(agg, _values(ref, metric, run_ids, ctx))
        _require_complete(ref, run_ids, ctx)
        rows = [ctx.rows[r] for r in run_ids]
        problem = unsound_aggregate(agg, rows, environment_prefix=ctx.environment_prefix)
        if problem:
            raise _RefError(f"{ref}: {problem}")
    except _RefError as exc:
        return str(exc)
    except (KeyError, ValueError) as exc:
        return f"{ref}: {exc}"
    if abs(float(format_value(expected, digits)) - float(shown)) > ctx.tolerance:
        return f"{ref} shows {shown} but the registry gives {format_value(expected, digits)}"
    return None


@dataclass(frozen=True)
class _Rules:
    ref: re.Pattern[str]
    ignore: tuple[re.Pattern[str], ...]
    escapes: tuple[re.Pattern[str], ...]
    numbers: tuple[re.Pattern[str], ...]
    marker: str
    # (prefix, suffix): text touching a tagged value that changes what the reader sees
    adjacent: tuple[re.Pattern[str], re.Pattern[str]]


def check_claims(config: Config) -> list[Finding]:
    root = config.root
    index = read_index(config.path("index_csv"))
    superseded = superseded_ids(index)
    ctx = _Context(
        rows={row["run_id"]: row for row in index},
        superseded=superseded,
        metric_prefix=str(config.get("registry.column_prefixes.metrics")),
        tolerance=config.float_value("claims.abs_tolerance"),
        superseded_is_error=config.bool_value("claims.superseded_is_error"),
        displayed=re.compile(config.str_value("claims.displayed_value_pattern")),
        current_by_cell=(
            current_runs_by_cell(index, superseded, config.str_value("registry.ok_status"))
            if config.bool_value("claims.require_complete_runs")
            else {}
        ),
        allow_pooled_cells=config.bool_value("tables.allow_pooled_cells"),
        environment_prefix=str(config.get("registry.column_prefixes.environment")),
    )
    adjacent = (
        re.compile(config.str_value("claims.value_prefix_pattern")),
        re.compile(config.str_value("claims.value_suffix_pattern")),
    )
    policy = Policy.from_config(config)
    hatch_categories = config.str_list("claims.escape_hatch_categories")
    numbers = tuple(
        re.compile(config.str_value(k))
        for k in ("claims.number_pattern", "claims.integer_result_pattern")
    )
    escapes = tuple(re.compile(p) for p in config.str_list("claims.escape_patterns"))
    marker = config.str_value("claims.ignore_line_marker")
    macro = re.escape(config.str_value("tables.macro"))
    groups = [
        (
            config.str_list("claims.tex_globs"),
            config.str_list("claims.strict_tex_globs"),
            _Rules(
                re.compile(rf"\\{macro}\{{(?P<ref>[^{{}}]*)\}}\{{(?P<value>[^{{}}]*)\}}"),
                tuple(
                    re.compile(p, re.MULTILINE)
                    for p in config.str_list("claims.tex_ignore_patterns")
                ),
                escapes,
                numbers,
                marker,
                adjacent,
            ),
        ),
        (
            config.str_list("claims.markdown_globs"),
            config.str_list("claims.strict_markdown_globs"),
            _Rules(
                re.compile(config.str_value("claims.markdown_ref_pattern")),
                tuple(re.compile(p) for p in config.str_list("claims.markdown_ignore_patterns")),
                escapes,
                numbers,
                marker,
                adjacent,
            ),
        ),
    ]
    findings: list[Finding] = []
    for globs, strict_globs, rules in groups:
        for path in iter_files(root, globs):
            relative = path.relative_to(root).as_posix()
            findings += _scan(
                path,
                rules,
                ctx,
                strict=first_match(relative, strict_globs) is not None,
                escapes_allowed=policy.in_categories(relative, hatch_categories),
            )
    _log.debug("claims check: %d finding(s)", len(findings))
    return findings


def _scan(
    path: Path, rules: _Rules, ctx: _Context, *, strict: bool, escapes_allowed: bool
) -> list[Finding]:
    findings: list[Finding] = []
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        for match in rules.ref.finditer(line):
            error = verify_reference(match.group("ref"), match.group("value"), ctx)
            if error:
                findings.append(Finding(path, lineno, "mismatch", error))
            prefix, suffix = rules.adjacent
            if prefix.search(line[: match.start()]) or suffix.match(line[match.end() :]):
                findings.append(
                    Finding(
                        path,
                        lineno,
                        "mismatch",
                        f"{match.group('ref')}: a sign or digit next to the tagged value "
                        "changes the number the reader sees",
                    )
                )
        if not strict or (escapes_allowed and rules.marker and rules.marker in line):
            continue
        residue = rules.ref.sub(" ", line)
        for pattern in (*rules.ignore, *(rules.escapes if escapes_allowed else ())):
            residue = pattern.sub(" ", residue)
        seen: set[tuple[int, int]] = set()
        for number in rules.numbers:
            for match in number.finditer(residue):
                if any(match.start() < e and s < match.end() for s, e in seen):
                    continue
                seen.add(match.span())
                findings.append(
                    Finding(
                        path,
                        lineno,
                        "untagged",
                        f"number {match.group(0).strip()} has no run reference",
                    )
                )
    return findings


def format_findings(findings: Sequence[Finding], root: Path) -> str:
    return "\n".join(f.render(root) for f in findings)
