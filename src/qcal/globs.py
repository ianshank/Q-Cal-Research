"""Root-anchored glob matching with ``**`` support, shared by policy and scanners.

Semantics: ``*`` and ``?`` never cross ``/``; ``**/`` matches zero or more whole
directories; a trailing ``**`` matches everything below. Patterns are anchored
at the repository root, so ``Makefile`` matches only the top-level file.

Stdlib-only: imported by Claude Code hooks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path


def _translate(pattern: str) -> str:
    text = pattern.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = text.lstrip("/")
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif text.startswith("**", i):
            out.append(".*")
            i += 2
        elif text[i] == "*":
            out.append("[^/]*")
            i += 1
        elif text[i] == "?":
            out.append("[^/]")
            i += 1
        elif text[i] == "[":
            close = text.find("]", i + 2 if text[i + 1 : i + 2] in {"!", "^"} else i + 1)
            if close == -1:
                out.append(re.escape("["))
                i += 1
                continue
            body = text[i + 1 : close]
            if body[:1] in {"!", "^"}:
                body = "^" + body[1:]
            out.append("[" + body.replace("\\", "\\\\") + "]")
            i = close + 1
        else:
            out.append(re.escape(text[i]))
            i += 1
    return "".join(out)


@lru_cache(maxsize=1024)
def compile_glob(pattern: str, *, case_insensitive: bool = True) -> re.Pattern[str]:
    flags = re.IGNORECASE if case_insensitive else 0
    return re.compile(_translate(pattern), flags | re.DOTALL)


def normalize(path: str) -> str:
    text = path.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def glob_match(path: str, pattern: str, *, case_insensitive: bool = True) -> bool:
    return (
        compile_glob(pattern, case_insensitive=case_insensitive).fullmatch(normalize(path))
        is not None
    )


def first_match(path: str, patterns: Iterable[str], *, case_insensitive: bool = True) -> str | None:
    """Return the first pattern that matches ``path``, or ``None``."""
    for pattern in patterns:
        if glob_match(path, pattern, case_insensitive=case_insensitive):
            return pattern
    return None


_GLOB_CHARS = frozenset("*?[")


def literal_base(pattern: str) -> tuple[str, bool]:
    """Leading directory of ``pattern`` with no glob characters, and whether the whole
    pattern is literal (a single path)."""
    parts = normalize(pattern).split("/")
    literal: list[str] = []
    for part in parts:
        if _GLOB_CHARS & set(part):
            return "/".join(literal), False
        literal.append(part)
    return "/".join(literal), True


def iter_files(
    root: Path, patterns: Sequence[str], *, case_insensitive: bool = False
) -> list[Path]:
    """All files under ``root`` matching any pattern, sorted and de-duplicated.

    Only the literal base directory of each pattern is walked, so ``paper/**/*.tex``
    never descends into ``.venv``.
    """
    if not patterns or not root.is_dir():
        return []
    found: set[Path] = set()
    for pattern in patterns:
        base, is_literal = literal_base(pattern)
        start = root / base if base else root
        if is_literal:
            if start.is_file():
                found.add(start)
            continue
        if not start.is_dir():
            continue
        for path in start.rglob("*"):
            if path.is_file() and glob_match(
                path.relative_to(root).as_posix(), pattern, case_insensitive=case_insensitive
            ):
                found.add(path)
    return sorted(found)
