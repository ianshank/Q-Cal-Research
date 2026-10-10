"""Path policy: which repository paths belong to which protection category.

Categories (``ian_only``, ``registry_only``, ``enforcement_surface`` by default)
are defined entirely in configuration, so new categories need no code change.
A path is checked twice, lexically and after resolving symlinks, so a symlink to
a protected file is still protected.

Stdlib-only: imported by Claude Code hooks.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from qcal.config import Config
from qcal.globs import first_match


@dataclass(frozen=True)
class PathVerdict:
    raw: str
    relative: str | None
    resolved_relative: str | None
    categories: tuple[str, ...]
    clean_room_hits: tuple[str, ...]

    @property
    def outside_root(self) -> bool:
        return self.relative is None and self.resolved_relative is None

    def blocked_by(self, deny: Iterable[str]) -> tuple[str, ...]:
        wanted = set(deny)
        return tuple(c for c in self.categories if c in wanted)


class Policy:
    def __init__(
        self,
        categories: Mapping[str, Sequence[str]],
        *,
        clean_room_substrings: Sequence[str] = (),
        case_insensitive: bool = True,
        messages: Mapping[str, str] | None = None,
    ) -> None:
        self._categories = {name: tuple(patterns) for name, patterns in categories.items()}
        self._clean_room = tuple(s for s in clean_room_substrings if s)
        self._ci = case_insensitive
        self._messages = dict(messages or {})

    @classmethod
    def from_config(cls, config: Config) -> Policy:
        raw = config.section("policy.categories")
        categories = {name: config.str_list(f"policy.categories.{name}") for name in raw}
        messages = config.get("policy.messages", {})
        return cls(
            categories,
            clean_room_substrings=config.str_list("policy.clean_room_substrings"),
            case_insensitive=config.bool_value("policy.case_insensitive"),
            messages={k: str(v) for k, v in messages.items()},
        )

    @property
    def category_names(self) -> tuple[str, ...]:
        return tuple(self._categories)

    def patterns(self, category: str) -> tuple[str, ...]:
        return self._categories.get(category, ())

    def categories_for(self, relative: str) -> tuple[str, ...]:
        return tuple(
            name
            for name, patterns in self._categories.items()
            if first_match(relative, patterns, case_insensitive=self._ci) is not None
        )

    def in_categories(self, relative: str, categories: Iterable[str]) -> bool:
        wanted = set(categories)
        return any(c in wanted for c in self.categories_for(relative))

    def clean_room_hits(self, text: str) -> tuple[str, ...]:
        haystack = text.lower() if self._ci else text
        return tuple(s for s in self._clean_room if (s.lower() if self._ci else s) in haystack)

    def message_for(self, category: str) -> str:
        return self._messages.get(category, f"is protected ({category}).")

    def evaluate(self, path: str | os.PathLike[str], root: Path) -> PathVerdict:
        raw = os.fspath(path)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        lexical = Path(os.path.normpath(candidate))
        resolved = Path(os.path.realpath(candidate))
        rel_lexical = _relative_to(lexical, Path(os.path.normpath(root)))
        rel_resolved = _relative_to(resolved, Path(os.path.realpath(root)))
        categories: list[str] = []
        for rel in (rel_lexical, rel_resolved):
            if rel is None:
                continue
            for category in self.categories_for(rel):
                if category not in categories:
                    categories.append(category)
        hits: list[str] = []
        for text in (raw, str(resolved)):
            for hit in self.clean_room_hits(text):
                if hit not in hits:
                    hits.append(hit)
        return PathVerdict(raw, rel_lexical, rel_resolved, tuple(categories), tuple(hits))


def _relative_to(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    text = relative.as_posix()
    return "" if text == "." else text
