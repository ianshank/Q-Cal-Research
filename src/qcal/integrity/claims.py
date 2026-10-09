"""Verify that every result number in the paper and claim documents traces to the registry.

Two kinds of finding:

* a tagged value whose reference does not resolve, references a superseded run,
  or does not equal the registry value at the displayed precision;
* an untagged decimal in a *strict* file (paper sections and tables, CLAIMS.md),
  after removing configured layout, citation and version contexts.

Lines carrying the configured ignore marker (``qcal:ignore``) are skipped, and
``\\qcalfixed{...}`` declares a non-result constant explicitly.

Stdlib-only: imported by the Claude Code Stop hook.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from qcal.config import Config
from qcal.globs import first_match, iter_files
from qcal.integrity.aggregates import aggregate, format_value
from qcal.log import get_logger
from qcal.registry.index import read_index

_log = get_logger("integrity.claims")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


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


class _RefError(ValueError):
    pass


def _parse_ref(ref: str) -> tuple[str, str, list[str]]:
    parts = ref.split(":")
    if parts[0] == "run" and len(parts) == 3:
        return "mean", parts[2], [parts[1]]
    if parts[0] == "agg" and len(parts) == 4:
        return parts[1], parts[2], [r for r in parts[3].split("+") if r]
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


def verify_reference(ref: str, displayed: str, ctx: _Context) -> str | None:
    """Return an error message, or ``None`` when ``displayed`` matches the registry."""
    number = _NUMBER.search(displayed)
    if number is None:
        return f"value {displayed!r} for {ref} is not a number"
    shown = number.group(0)
    digits = len(shown.split(".", 1)[1]) if "." in shown else 0
    try:
        agg, metric, run_ids = _parse_ref(ref)
        expected = aggregate(agg, _values(ref, metric, run_ids, ctx))
    except _RefError as exc:
        return str(exc)
    except (KeyError, ValueError) as exc:
        return f"{ref}: {exc}"
    if abs(float(format_value(expected, digits)) - float(shown)) > ctx.tolerance:
        return f"{ref} shows {shown} but the registry gives {format_value(expected, digits)}"
    return None


def check_claims(config: Config) -> list[Finding]:
    root = config.root
    index = read_index(config.path("index_csv"))
    ctx = _Context(
        rows={row["run_id"]: row for row in index},
        superseded=frozenset(r["supersedes"] for r in index if r.get("supersedes")),
        metric_prefix=str(config.get("registry.column_prefixes.metrics")),
        tolerance=config.float_value("claims.abs_tolerance"),
        superseded_is_error=config.bool_value("claims.superseded_is_error"),
    )
    marker = config.str_value("claims.ignore_line_marker")
    number = re.compile(config.str_value("claims.number_pattern"))
    findings: list[Finding] = []

    macro = re.escape(config.str_value("tables.macro"))
    tex_ref = re.compile(rf"\\{macro}\{{(?P<ref>[^{{}}]*)\}}\{{(?P<value>[^{{}}]*)\}}")
    tex_ignore = [
        re.compile(p, re.MULTILINE) for p in config.str_list("claims.tex_ignore_patterns")
    ]
    strict_tex = config.str_list("claims.strict_tex_globs")
    for path in iter_files(root, config.str_list("claims.tex_globs")):
        strict = first_match(path.relative_to(root).as_posix(), strict_tex) is not None
        findings += _scan(path, tex_ref, tex_ignore, number, ctx, strict=strict, marker=marker)

    md_ref = re.compile(config.str_value("claims.markdown_ref_pattern"))
    md_ignore = [re.compile(p) for p in config.str_list("claims.markdown_ignore_patterns")]
    strict_md = config.str_list("claims.strict_markdown_globs")
    for path in iter_files(root, config.str_list("claims.markdown_globs")):
        strict = first_match(path.relative_to(root).as_posix(), strict_md) is not None
        findings += _scan(path, md_ref, md_ignore, number, ctx, strict=strict, marker=marker)

    _log.debug("claims check: %d finding(s)", len(findings))
    return findings


def _scan(
    path: Path,
    ref_pattern: re.Pattern[str],
    ignore: Sequence[re.Pattern[str]],
    number: re.Pattern[str],
    ctx: _Context,
    *,
    strict: bool,
    marker: str,
) -> list[Finding]:
    findings: list[Finding] = []
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if marker and marker in line:
            continue
        for match in ref_pattern.finditer(line):
            error = verify_reference(match.group("ref"), match.group("value"), ctx)
            if error:
                findings.append(Finding(path, lineno, "mismatch", error))
        if not strict:
            continue
        residue = ref_pattern.sub(" ", line)
        for pattern in ignore:
            residue = pattern.sub(" ", residue)
        for match in number.finditer(residue):
            findings.append(
                Finding(path, lineno, "untagged", f"number {match.group(0)} has no run reference")
            )
    return findings


def format_findings(findings: Sequence[Finding], root: Path) -> str:
    return "\n".join(f.render(root) for f in findings)
