"""Validation of ``.claude/settings.json`` hooks and ``.mcp.json`` servers.

Claude Code ignores unknown hook events and silently never fires a hook whose
matcher names no real tool, so both are checked against configured lists.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qcal.config import Config

_PROJECT_DIR_REF = re.compile(r'"?\$\{?CLAUDE_PROJECT_DIR\}?"?(?P<rest>/[^\s"]+)')
_REMOTE_KEYS = ("url",)
_IDENTIFIER_ALTERNATION = re.compile(r"^[A-Za-z_][\w]*(?:\|[A-Za-z_][\w]*)*$")

Report = Callable[[str], None]


def _load_json(path: Path, shown: Path, report: Report) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        report(f"{shown}: invalid JSON ({exc})")
        return None
    if not isinstance(data, dict):
        report(f"{shown}: top level must be a JSON object")
        return None
    return data


def _check_matcher(config: Config, event: str, matcher: Any, shown: Path, report: Report) -> None:
    if matcher in (None, "", "*"):
        return
    if not isinstance(matcher, str):
        report(f"{shown}: {event} matcher must be a string, got {type(matcher).__name__}")
        return
    try:
        re.compile(matcher)
    except re.error as exc:
        report(f"{shown}: {event} matcher {matcher!r} is not a valid regex ({exc})")
        return
    if not _IDENTIFIER_ALTERNATION.match(matcher):
        return  # a real regex (for example mcp__.*): cannot be checked name by name
    names = matcher.split("|")
    if event in config.str_list("agent_layer.tool_matcher_events"):
        known = set(config.str_list("agent_layer.known_tools"))
        prefixes = tuple(config.str_list("agent_layer.network_tool_prefixes"))
        unknown = [n for n in names if n not in known and not n.startswith(prefixes)]
        if unknown:
            report(f"{shown}: {event} matcher names unknown tools {unknown}")
    elif event == "SessionStart":
        sources = set(config.str_list("agent_layer.session_start_sources"))
        unknown = [n for n in names if n not in sources]
        if unknown:
            report(f"{shown}: SessionStart matcher names unknown sources {unknown}")


def _hook_names() -> set[str]:
    from qcal.hooks.cli import build_parser

    for action in build_parser()._actions:
        if action.choices is not None and isinstance(action.choices, dict):
            return set(action.choices)
    return set()  # pragma: no cover - the hook parser always has subcommands


def wrapped_hook(config: Config, command: str) -> str | None:
    """The qcal hook a settings command runs through the wrapper, or ``None``."""
    wrapper = config.str_value("agent_layer.hook_wrapper")
    words = command.replace('"', " ").split()
    for i, word in enumerate(words):
        if word.endswith(wrapper):
            rest = [w for w in words[i + 1 :] if w != "--fail-open"]
            return rest[0] if rest else ""
    return None


def _check_command(
    config: Config, event: str, command: str, shown: Path, report: Report
) -> str | None:
    """Validate one command hook; returns the wrapped qcal hook name, if any."""
    root = config.root
    match = _PROJECT_DIR_REF.search(command)
    if not match:
        report(
            f"{shown}: {event} hook {command!r} should locate its script via $CLAUDE_PROJECT_DIR"
        )
        return None
    script = (root / match.group("rest").lstrip("/")).resolve()
    if not script.is_relative_to(root.resolve()):
        report(f"{shown}: {event} hook {command!r} points outside the project")
        return None
    script_shown = script.relative_to(root.resolve())
    if not script.is_file():
        report(f"{shown}: {event} hook script {script_shown} is missing")
    elif not os.access(script, os.X_OK):
        report(f"{shown}: {event} hook script {script_shown} is not executable")
    hook = wrapped_hook(config, command)
    if hook is not None and hook not in _hook_names():
        report(f"{shown}: {event} hook {command!r} runs unknown qcal hook {hook!r}")
    return hook


def _check_guard_matchers(
    config: Config, event: str, matcher: Any, *, hooks: list[str], shown: Path, report: Report
) -> None:
    """The guards must see every tool they exist to judge, or they silently never run."""
    names = set(matcher.split("|")) if isinstance(matcher, str) else set()
    wildcard = matcher in (None, "", "*")
    required = {
        "guard-paths": set(config.str_list("hooks.edit_tools")),
        "guard-bash": {"Bash"},
    }
    for hook in hooks:
        missing = sorted(required.get(hook, set()) - names)
        if event == "PreToolUse" and missing and not wildcard:
            report(f"{shown}: {event} matcher {matcher!r} for {hook} misses tools {missing}")


def check_settings(config: Config, report: Report) -> int:
    """Validate hooks in the settings file; returns the number of command hooks."""
    shown = Path(config.str_value("agent_layer.settings_file"))
    settings = _load_json(config.root / shown, shown, report)
    if settings is None:
        return 0
    hooks = settings.get("hooks") or {}
    _check_required_hooks(config, hooks, shown, report)
    return check_hooks(config, hooks, shown, report)


def _check_required_hooks(config: Config, hooks: Any, shown: Path, report: Report) -> None:
    """The guards must be wired as PreToolUse hooks; elsewhere they would never block."""
    if not isinstance(hooks, dict):
        return
    wired = {
        wrapped_hook(config, str(hook.get("command", "")))
        for group in hooks.get("PreToolUse") or []
        if isinstance(group, dict)
        for hook in group.get("hooks", [])
        if isinstance(hook, dict)
    }
    for required in config.str_list("agent_layer.required_pretooluse_hooks"):
        if required not in wired:
            report(f"{shown}: required PreToolUse hook {required!r} is not wired")


def check_hooks(config: Config, hooks: Any, shown: Path, report: Report) -> int:
    """Validate one ``hooks`` mapping (settings file or agent frontmatter)."""
    if not isinstance(hooks, dict):
        report(f"{shown}: hooks must be a mapping of event names to matcher groups")
        return 0
    events = set(config.str_list("agent_layer.hook_events"))
    count = 0
    for event, groups in hooks.items():
        if event not in events:
            report(f"{shown}: unknown hook event {event!r} (Claude Code would ignore it)")
        if not isinstance(groups, list):
            report(f"{shown}: {event} must be a list of matcher groups")
            continue
        for group in groups:
            if not isinstance(group, dict):
                report(f"{shown}: {event} matcher groups must be mappings")
                continue
            _check_matcher(config, event, group.get("matcher"), shown, report)
            wrapped: list[str] = []
            for hook in group.get("hooks", []):
                if hook.get("type", "command") != "command":
                    continue  # prompt/agent/http hooks have no script to locate
                count += 1
                name = _check_command(config, event, hook.get("command", ""), shown, report)
                if name:
                    wrapped.append(name)
            _check_guard_matchers(
                config, event, group.get("matcher"), hooks=wrapped, shown=shown, report=report
            )
    return count


def check_mcp(config: Config, report: Report) -> int:
    """Validate ``.mcp.json``; returns the number of servers."""
    shown = Path(config.str_value("agent_layer.mcp_file"))
    data = _load_json(config.root / shown, shown, report)
    if data is None:
        return 0
    servers = data.get("mcpServers", {})
    for name, server in servers.items():
        if any(k in server for k in _REMOTE_KEYS) and "type" not in server:
            report(f"{shown}: remote server {name!r} needs an explicit 'type'")
        for value in (server.get("headers") or {}).values():
            if isinstance(value, str) and value.lower().startswith("bearer ") and "${" not in value:
                report(f"{shown}: server {name!r} embeds a literal credential")
    return len(servers)


__all__ = ["check_hooks", "check_mcp", "check_settings", "wrapped_hook"]
