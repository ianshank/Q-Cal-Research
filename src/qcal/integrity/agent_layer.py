"""Static validation of the Claude Code agent layer.

Claude Code silently ignores unknown frontmatter keys, so a typo in a subagent
file fails quietly (plan §3.7). This check makes it fail loudly, and also checks
that every hook command in settings points at a script that exists.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.config import Config
from qcal.integrity.licenses import FrontmatterError, read_frontmatter
from qcal.log import get_logger

_log = get_logger("integrity.agent_layer")
_PROJECT_DIR_REF = re.compile(r'"?\$\{?CLAUDE_PROJECT_DIR\}?"?(?P<rest>/[^\s"]+)')
_REMOTE_KEYS = ("url",)


@dataclass
class AgentLayerReport:
    errors: list[str] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": "PASS" if self.passed else "FAIL",
            "errors": self.errors,
            "checked": self.checked,
        }


def _check_frontmatter(
    path: Path,
    data: dict[str, Any],
    *,
    allowed: set[str],
    required: list[str],
    report: AgentLayerReport,
    root: Path,
) -> None:
    shown = path.relative_to(root)
    if not data:
        report.errors.append(f"{shown}: missing or empty YAML frontmatter")
        return
    unknown = sorted(set(data) - allowed)
    if unknown:
        report.errors.append(
            f"{shown}: unknown frontmatter keys {unknown} (Claude Code would ignore them)"
        )
    missing = [k for k in required if not data.get(k)]
    if missing:
        report.errors.append(f"{shown}: missing required keys {missing}")


def check_agent_layer(config: Config) -> AgentLayerReport:
    root = config.root
    report = AgentLayerReport()
    forbidden_modes = set(config.str_list("agent_layer.forbidden_permission_modes"))

    agents = sorted(root.glob(config.str_value("agent_layer.agents_glob")))
    agent_allowed = set(config.str_list("agent_layer.agent_allowed_keys"))
    agent_required = config.str_list("agent_layer.agent_required_keys")
    for path in agents:
        data = _frontmatter(path, root, report)
        if data is None:
            continue
        _check_frontmatter(
            path, data, allowed=agent_allowed, required=agent_required, report=report, root=root
        )
        if data.get("name") and data["name"] != path.stem:
            report.errors.append(f"{path.relative_to(root)}: name {data['name']!r} != file name")
        if data.get("permissionMode") in forbidden_modes:
            report.errors.append(
                f"{path.relative_to(root)}: permissionMode {data['permissionMode']} is forbidden"
            )
    report.checked["agents"] = len(agents)

    skills = sorted(root.glob(config.str_value("agent_layer.skills_glob")))
    skill_allowed = set(config.str_list("agent_layer.skill_allowed_keys"))
    for path in skills:
        data = _frontmatter(path, root, report)
        if data is None:
            continue
        _check_frontmatter(
            path, data, allowed=skill_allowed, required=["description"], report=report, root=root
        )
    report.checked["skills"] = len(skills)

    report.checked["hook_commands"] = _check_settings(
        root, Path(config.str_value("agent_layer.settings_file")), report
    )
    report.checked["mcp_servers"] = _check_mcp(
        root, Path(config.str_value("agent_layer.mcp_file")), report
    )
    _log.info("agent layer: %s", "PASS" if report.passed else "FAIL")
    return report


def _frontmatter(path: Path, root: Path, report: AgentLayerReport) -> dict[str, Any] | None:
    try:
        return read_frontmatter(path.read_text("utf-8"))
    except FrontmatterError as exc:
        report.errors.append(f"{path.relative_to(root)}: {exc}")
        return None


def _check_settings(root: Path, settings_file: Path, report: AgentLayerReport) -> int:
    path = root / settings_file
    if not path.is_file():
        return 0
    try:
        settings = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        report.errors.append(f"{settings_file}: invalid JSON ({exc})")
        return 0
    count = 0
    for event, groups in (settings.get("hooks") or {}).items():
        for group in groups:
            for hook in group.get("hooks", []):
                if hook.get("type", "command") != "command":
                    continue  # prompt/agent/http hooks have no script to locate
                command = hook.get("command", "")
                count += 1
                match = _PROJECT_DIR_REF.search(command)
                if not match:
                    report.errors.append(
                        f"{settings_file}: {event} hook {command!r} should locate its script "
                        "via $CLAUDE_PROJECT_DIR"
                    )
                    continue
                script = root / match.group("rest").lstrip("/")
                shown = script.relative_to(root)
                if not script.is_file():
                    report.errors.append(f"{settings_file}: {event} hook script {shown} is missing")
                elif not os.access(script, os.X_OK):
                    report.errors.append(
                        f"{settings_file}: {event} hook script {shown} is not executable"
                    )
    return count


def _check_mcp(root: Path, mcp_file: Path, report: AgentLayerReport) -> int:
    path = root / mcp_file
    if not path.is_file():
        return 0
    try:
        servers = json.loads(path.read_text("utf-8")).get("mcpServers", {})
    except json.JSONDecodeError as exc:
        report.errors.append(f"{mcp_file}: invalid JSON ({exc})")
        return 0
    for name, server in servers.items():
        if any(k in server for k in _REMOTE_KEYS) and "type" not in server:
            report.errors.append(f"{mcp_file}: remote server {name!r} needs an explicit 'type'")
        for value in (server.get("headers") or {}).values():
            if isinstance(value, str) and value.lower().startswith("bearer ") and "${" not in value:
                report.errors.append(f"{mcp_file}: server {name!r} embeds a literal credential")
    return len(servers)
