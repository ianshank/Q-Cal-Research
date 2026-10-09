"""Parsing of the JSON document Claude Code writes to a hook's stdin."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class PayloadError(ValueError):
    """The hook input was not a JSON object."""


@dataclass(frozen=True)
class HookPayload:
    hook_event_name: str | None = None
    tool_name: str | None = None
    tool_input: Mapping[str, Any] = field(default_factory=dict)
    cwd: str | None = None
    session_id: str | None = None
    stop_hook_active: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> HookPayload:
        try:
            data = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            raise PayloadError(f"hook input is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise PayloadError("hook input must be a JSON object")
        tool_input = data.get("tool_input")
        return cls(
            hook_event_name=_opt_str(data.get("hook_event_name")),
            tool_name=_opt_str(data.get("tool_name")),
            tool_input=tool_input if isinstance(tool_input, dict) else {},
            cwd=_opt_str(data.get("cwd")),
            session_id=_opt_str(data.get("session_id")),
            stop_hook_active=data.get("stop_hook_active") is True,
            raw=data,
        )

    def string_fields(self, names: list[str]) -> list[str]:
        """Non-empty string values of ``tool_input`` fields named in ``names``, in order."""
        values: list[str] = []
        for name in names:
            value = self.tool_input.get(name)
            if isinstance(value, str) and value and value not in values:
                values.append(value)
        return values


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None
