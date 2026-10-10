"""Documented commands must exist.

Skills, agents, CLAUDE.md, the Makefile and workflows tell people and agents which
``qcal`` commands and ``make`` targets to run. When a command is renamed those
instructions rot silently, and an agent following them fails or improvises. This
check reads every configured file, finds command references in *code* (fenced
blocks and inline code in Markdown; every line elsewhere) and resolves them against
the real argument parser and the Makefile's targets.

It validates the command path and option names only, never argument values, so
placeholders such as ``<id>`` or ``$BASE`` are fine.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from qcal.config import Config
from qcal.globs import first_match, iter_files
from qcal.log import get_logger

_log = get_logger("integrity.command_refs")

# Invocations of the CLI. The console scripts map onto a subcommand prefix.
_QCAL_INVOCATION = re.compile(
    r"(?<![\w./$'\"-])(?P<cmd>\$\(QCAL\)|-m\s+qcal|qcal-registry|qcal-claims|qcal)(?=\s|$)"
)
_SCRIPT_PREFIX = {"qcal-registry": ("registry",), "qcal-claims": ("claims",)}
_MAKE_INVOCATION = re.compile(r"(?<![\w./$'\"-])(?:make|\$\(MAKE\))(?=\s|$)")
_MAKE_RULE = re.compile(
    r"^(?P<targets>[A-Za-z0-9][\w./-]*(?:[ \t]+[A-Za-z0-9][\w./-]*)*)\s*:(?![=:])"
)
# Shell syntax that ends a command: pipes, lists, redirections, subshells, comments.
_COMMAND_END = re.compile(r"[|;&<>()`#]")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Outside Markdown: comments, and YAML keys that hold prose rather than commands.
_COMMENT = re.compile(r"(^|\s)#.*$")
_YAML_PROSE_KEY = re.compile(r"^\s*(?:-\s*)?(?:name|description|title)\s*:")
_YAML_SUFFIXES = (".yml", ".yaml")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_PLACEHOLDER_START = ("<", "{", "$", "[", "...", "…")


@dataclass(frozen=True)
class CommandRef:
    file: str
    line: int
    kind: str  # "qcal" | "make"
    tokens: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join((self.kind, *self.tokens))


@dataclass(frozen=True)
class CommandRefError:
    ref: CommandRef
    problem: str

    def __str__(self) -> str:
        return f"{self.ref.file}:{self.ref.line}: `{self.ref.text}` {self.problem}"


def code_lines(text: str, *, markdown: bool, yaml: bool = False) -> Iterator[tuple[int, str]]:
    """Lines (1-based) holding code.

    Markdown: fenced blocks and inline code spans. Everything else: every line with
    ``#`` comments removed and, for YAML, prose keys such as ``name:`` skipped.
    """
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if not markdown:
            if not (yaml and _YAML_PROSE_KEY.match(line)):
                yield number, _COMMENT.sub("", line)
            continue
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            yield number, line
            continue
        for span in _INLINE_CODE.finditer(line):
            yield number, span.group(1)


def _tokens(rest: str) -> tuple[str, ...]:
    end = _COMMAND_END.search(rest)
    words = (rest[: end.start()] if end else rest).split()
    # A trailing backslash continues a shell line; it is never an argument.
    return tuple(word for word in (w.strip("'\"") for w in words) if word and word != "\\")


def extract_refs(text: str, file: str, *, markdown: bool) -> list[CommandRef]:
    """Every ``qcal`` and ``make`` invocation in the code parts of ``text``."""
    refs: list[CommandRef] = []
    yaml = file.endswith(_YAML_SUFFIXES)
    for number, line in code_lines(text, markdown=markdown, yaml=yaml):
        for match in _QCAL_INVOCATION.finditer(line):
            prefix = _SCRIPT_PREFIX.get(match.group("cmd"), ())
            refs.append(CommandRef(file, number, "qcal", prefix + _tokens(line[match.end() :])))
        refs.extend(
            CommandRef(file, number, "make", _tokens(line[match.end() :]))
            for match in _MAKE_INVOCATION.finditer(line)
        )
    return refs


def _is_placeholder(token: str) -> bool:
    return token.startswith(_PLACEHOLDER_START)


# argparse has no public introspection API; ``_actions`` is stable across 3.11-3.13.
def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser] | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return None


def _options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {opt: action for action in parser._actions for opt in action.option_strings}


def _takes_rest(parser: argparse.ArgumentParser) -> bool:
    return any(action.nargs == argparse.REMAINDER for action in parser._actions)


def validate_qcal(parser: argparse.ArgumentParser, tokens: Sequence[str]) -> str | None:
    """``None`` when ``tokens`` name an existing command path with existing options."""
    current, path = parser, ["qcal"]
    skip_value = False
    for word in tokens:
        if skip_value:
            skip_value = False
            continue
        if word == "--" or _takes_rest(current):
            return None
        if word.startswith("-") and len(word) > 1:
            name, has_value = word.split("=", 1)[0], "=" in word
            action = _options(current).get(name)
            if action is None:
                return f"has no option {name} (in `{' '.join(path)}`)"
            skip_value = action.nargs != 0 and not has_value
            continue
        subcommands = _subcommands(current)
        if subcommands is None:
            continue  # a positional value
        if _is_placeholder(word):
            return None
        if word not in subcommands:
            return f"has no command {word!r} (in `{' '.join(path)}`)"
        current = subcommands[word]
        path.append(word)
    return None


def make_targets(makefile: Path) -> set[str]:
    """Explicit rule targets in ``makefile`` (pattern and special targets excluded)."""
    targets: set[str] = set()
    if not makefile.is_file():
        return targets
    for line in makefile.read_text("utf-8").splitlines():
        match = _MAKE_RULE.match(line)
        if match:
            targets.update(match.group("targets").split())
    return targets


def validate_make(targets: set[str], tokens: Sequence[str]) -> str | None:
    for word in tokens:
        if word.startswith("-") or "=" in word or _is_placeholder(word):
            continue
        if word not in targets:
            return f"has no Makefile target {word!r}"
    return None


def _default_parser() -> argparse.ArgumentParser:
    from qcal.cli.app import build_parser

    return build_parser()


def check_command_refs(
    config: Config,
    *,
    parser_factory: Callable[[], argparse.ArgumentParser] = _default_parser,
) -> tuple[list[CommandRefError], int]:
    """All drifted references, and how many references were checked."""
    root = config.root
    exclude = config.str_list("agent_layer.command_ref_exclude_globs")
    markdown_globs = config.str_list("agent_layer.command_ref_markdown_globs")
    parser = parser_factory()
    targets = make_targets(root / config.str_value("agent_layer.makefile"))
    errors: list[CommandRefError] = []
    checked = 0
    for path in iter_files(root, config.str_list("agent_layer.command_ref_globs")):
        relative = path.relative_to(root).as_posix()
        if first_match(relative, exclude) is not None:
            continue
        try:
            text = path.read_text("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _log.warning("cannot read %s for command references: %s", relative, exc)
            continue
        markdown = first_match(relative, markdown_globs) is not None
        for ref in extract_refs(text, relative, markdown=markdown):
            checked += 1
            if ref.kind == "qcal":
                problem = validate_qcal(parser, ref.tokens)
            else:
                problem = validate_make(targets, ref.tokens) if targets else None
            if problem:
                errors.append(CommandRefError(ref, problem))
    _log.debug("checked %d command reference(s); %d drifted", checked, len(errors))
    return errors, checked


__all__ = [
    "CommandRef",
    "CommandRefError",
    "check_command_refs",
    "code_lines",
    "extract_refs",
    "make_targets",
    "validate_make",
    "validate_qcal",
]
