"""The shape every check report shares, so the CLI renders and exits uniformly.

Reports stay separate dataclasses (their fields differ), but each exposes
``passed``, ``to_dict()`` (JSON, with a ``verdict``) and ``render_text()`` (humans).
"""

from __future__ import annotations

from typing import Any, Final, Protocol, runtime_checkable

PASS: Final = "PASS"  # noqa: S105 - a verdict label, not a secret
FAIL: Final = "FAIL"
SKIP: Final = "SKIP"


def verdict(passed: bool) -> str:  # noqa: FBT001 - a predicate, not a flag
    return PASS if passed else FAIL


@runtime_checkable
class CheckReport(Protocol):
    @property
    def passed(self) -> bool: ...

    def to_dict(self) -> dict[str, Any]: ...

    def render_text(self) -> str: ...


__all__ = ["FAIL", "PASS", "SKIP", "CheckReport", "verdict"]
