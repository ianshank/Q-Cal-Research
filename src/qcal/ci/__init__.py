"""CI integrity checks that read only git objects and therefore can run safely under
``pull_request_target`` with the *base* branch's code and policy."""

from __future__ import annotations

from typing import Final

from qcal.config import ConfigError

#: ``signing.mode`` and ``review.mode``: report only, or fail the check.
MODES: Final = ("bootstrap", "enforce")


def check_mode(value: str, key: str) -> str:
    """``value`` if it is one of :data:`MODES`; a misspelt mode is a configuration error."""
    if value not in MODES:
        raise ConfigError(f"{key} must be one of {MODES}, got {value!r}")
    return value


__all__ = ["MODES", "check_mode"]
