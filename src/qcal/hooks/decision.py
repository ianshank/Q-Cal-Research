"""Hook decisions, the human-controlled guard mode, and the decision log."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, Final

from qcal.config import Config, ConfigError
from qcal.log import get_logger

_log = get_logger("hooks")

#: Claude Code protocol: exit code 2 blocks the tool call (any other non-zero does not).
BLOCK_EXIT_CODE: Final = 2


class Mode(StrEnum):
    ENFORCE = "enforce"
    WARN = "warn"
    OFF = "off"


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str = ""
    rule: str = ""

    @classmethod
    def allowed(cls, rule: str = "") -> Decision:
        return cls(allow=True, reason="", rule=rule)

    @classmethod
    def denied(cls, reason: str, rule: str) -> Decision:
        return cls(allow=False, reason=reason, rule=rule)


def resolve_mode(config: Config, environ: Mapping[str, str]) -> Mode:
    """Mode comes from the environment Claude Code was launched with, so agents cannot flip it."""
    raw = environ.get(config.str_value("hooks.mode_env")) or config.str_value("hooks.default_mode")
    try:
        return Mode(raw.strip().lower())
    except ValueError:
        expected = [m.value for m in Mode]
        raise ConfigError(f"unknown guard mode {raw!r}; expected one of {expected}") from None


def emit(
    decision: Decision,
    *,
    hook: str,
    mode: Mode,
    config: Config,
    stderr: IO[str],
    environ: Mapping[str, str],
    payload: Mapping[str, Any] | None = None,
) -> int:
    """Report a decision and return the process exit code Claude Code expects."""
    _record(decision, hook=hook, mode=mode, config=config, environ=environ, payload=payload)
    if decision.allow or mode is Mode.OFF:
        return 0
    if mode is Mode.WARN:
        stderr.write(f"WARNING (qcal guard in warn mode): {decision.reason}\n")
        return 0
    stderr.write(f"BLOCKED: {decision.reason}\n")
    return BLOCK_EXIT_CODE


def _record(
    decision: Decision,
    *,
    hook: str,
    mode: Mode,
    config: Config,
    environ: Mapping[str, str],
    payload: Mapping[str, Any] | None,
) -> None:
    _log.debug(
        "hook=%s allow=%s rule=%s reason=%s", hook, decision.allow, decision.rule, decision.reason
    )
    target = environ.get(config.str_value("hooks.log_file_env")) or config.str_value(
        "hooks.log_file"
    )
    if not target:
        return
    path = Path(target)
    if not path.is_absolute():
        path = config.root / path
    entry = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "hook": hook,
        "mode": mode.value,
        "allow": decision.allow,
        "rule": decision.rule,
        "reason": decision.reason,
        "tool": (payload or {}).get("tool_name"),
        "session": (payload or {}).get("session_id"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + os.linesep)
    except OSError as exc:  # logging must never change a decision
        _log.warning("cannot write hook decision log %s: %s", path, exc)
