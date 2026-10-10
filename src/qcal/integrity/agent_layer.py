"""Static, deterministic validation of the Claude Code agent layer.

Claude Code silently ignores unknown frontmatter keys, unknown hook events and
matchers that name no tool, so a typo fails quietly (plan §3.7). This check makes
it fail loudly. It covers:

* agent and skill frontmatter: known keys, required keys, file-name agreement;
* tool policy: agents declare their tools; no agent or skill combines a
  write-capable tool with a network tool; models come from an allowlist;
* references: ``context: fork`` skills name a real agent, agents name real skills;
* hooks and MCP servers (:mod:`qcal.integrity.agent_settings`);
* command-reference drift (:mod:`qcal.integrity.command_refs`).

Every list it checks against lives in ``[agent_layer]`` configuration.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.config import Config
from qcal.integrity.agent_settings import check_hooks, check_mcp, check_settings
from qcal.integrity.command_refs import check_command_refs
from qcal.integrity.licenses import FrontmatterError, read_frontmatter
from qcal.log import get_logger
from qcal.reports import verdict

_log = get_logger("integrity.agent_layer")
_TOOL_NAME = re.compile(r"^\s*(?P<name>[\w-]+)")


@dataclass
class AgentLayerReport:
    errors: list[str] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": verdict(self.passed),
            "errors": self.errors,
            "checked": self.checked,
        }

    def render_text(self) -> str:
        return "\n".join(
            [
                f"agent layer: {verdict(self.passed)} {self.checked}",
                *(f"  error: {e}" for e in self.errors),
            ]
        )


def tool_names(value: Any) -> list[str]:
    """Tool names from a frontmatter value: ``"Read, Bash(git:*)"`` or a YAML list."""
    items: Iterable[Any]
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else value
    names = []
    for item in items:
        match = _TOOL_NAME.match(str(item))
        if match:
            names.append(match.group("name"))
    return names


def _raw_tools(value: Any) -> list[str]:
    """The entries ``tool_names`` parses, unparsed (same order, same filtering)."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else value
    return [str(item).strip() for item in items if _TOOL_NAME.match(str(item))]


@dataclass(frozen=True)
class _ToolPolicy:
    write: frozenset[str]
    network: frozenset[str]
    network_prefixes: tuple[str, ...]
    models: frozenset[str]
    model_pattern: re.Pattern[str]

    @classmethod
    def from_config(cls, config: Config) -> _ToolPolicy:
        return cls(
            write=frozenset(config.str_list("agent_layer.write_tools")),
            network=frozenset(config.str_list("agent_layer.network_tools")),
            network_prefixes=tuple(config.str_list("agent_layer.network_tool_prefixes")),
            models=frozenset(config.str_list("agent_layer.allowed_models")),
            model_pattern=re.compile(config.str_value("agent_layer.model_id_pattern")),
        )

    def conflict(self, tools: Iterable[str]) -> tuple[list[str], list[str]] | None:
        present = set(tools)
        writes = sorted(present & self.write)
        network = sorted(
            t for t in present if t in self.network or t.startswith(self.network_prefixes)
        )
        return (writes, network) if writes and network else None

    def model_ok(self, model: Any) -> bool:
        return isinstance(model, str) and (
            model in self.models or self.model_pattern.fullmatch(model) is not None
        )


def _check_frontmatter(
    shown: Path,
    data: dict[str, Any],
    *,
    allowed: set[str],
    required: list[str],
    report: AgentLayerReport,
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        report.errors.append(
            f"{shown}: unknown frontmatter keys {unknown} (Claude Code would ignore them)"
        )
    missing = [k for k in required if not data.get(k)]
    if missing:
        report.errors.append(f"{shown}: missing required keys {missing}")


def _check_tools(
    shown: Path, tools: list[str], policy: _ToolPolicy, report: AgentLayerReport
) -> None:
    conflict = policy.conflict(tools)
    if conflict:
        writes, network = conflict
        report.errors.append(
            f"{shown}: combines write-capable tools {writes} with network tools {network}; "
            "split the work across two agents"
        )


def _check_model(
    shown: Path, data: dict[str, Any], policy: _ToolPolicy, report: AgentLayerReport
) -> None:
    if "model" in data and not policy.model_ok(data["model"]):
        report.errors.append(f"{shown}: model {data['model']!r} is not in the allowlist")


def _check_bool_keys(
    shown: Path, data: dict[str, Any], keys: Iterable[str], report: AgentLayerReport
) -> None:
    for key in keys:
        if key in data and not isinstance(data[key], bool):
            report.errors.append(f"{shown}: {key} must be true or false, got {data[key]!r}")


def check_agent_layer(config: Config) -> AgentLayerReport:
    root = config.root
    report = AgentLayerReport()
    policy = _ToolPolicy.from_config(config)

    agent_files = sorted(root.glob(config.str_value("agent_layer.agents_glob")))
    skill_files = sorted(root.glob(config.str_value("agent_layer.skills_glob")))
    agents = _load_all(root, agent_files, report)
    skills = _load_all(root, skill_files, report)
    agent_names = {path.stem for path in agents} | set(
        config.str_list("agent_layer.builtin_agents")
    )
    skill_names = {path.parent.name for path in skills}

    _check_agents(
        config, root=root, agents=agents, skill_names=skill_names, policy=policy, report=report
    )
    _check_skills(
        config, root=root, skills=skills, agent_names=agent_names, policy=policy, report=report
    )
    report.checked["agents"] = len(agent_files)
    report.checked["skills"] = len(skill_files)
    report.checked["hook_commands"] = check_settings(config, report.errors.append)
    report.checked["mcp_servers"] = check_mcp(config, report.errors.append)
    drift, checked = check_command_refs(config)
    report.errors.extend(f"command drift: {error}" for error in drift)
    report.checked["command_refs"] = checked
    _log.info("agent layer: %s %s", verdict(report.passed), report.checked)
    return report


def _load_all(
    root: Path, paths: Iterable[Path], report: AgentLayerReport
) -> dict[Path, dict[str, Any]]:
    loaded: dict[Path, dict[str, Any]] = {}
    for path in paths:
        shown = path.relative_to(root)
        try:
            data = read_frontmatter(path.read_text("utf-8"))
        except FrontmatterError as exc:
            report.errors.append(f"{shown}: {exc}")
            continue
        if not data:
            report.errors.append(f"{shown}: missing or empty YAML frontmatter")
            continue
        loaded[path] = data
    return loaded


def _check_agents(
    config: Config,
    *,
    root: Path,
    agents: dict[Path, dict[str, Any]],
    skill_names: set[str],
    policy: _ToolPolicy,
    report: AgentLayerReport,
) -> None:
    allowed = set(config.str_list("agent_layer.agent_allowed_keys"))
    required = config.str_list("agent_layer.agent_required_keys")
    forbidden_modes = set(config.str_list("agent_layer.forbidden_permission_modes"))
    for path, data in agents.items():
        shown = path.relative_to(root)
        _check_frontmatter(shown, data, allowed=allowed, required=required, report=report)
        if data.get("name") and data["name"] != path.stem:
            report.errors.append(f"{shown}: name {data['name']!r} != file name")
        if data.get("permissionMode") in forbidden_modes:
            report.errors.append(f"{shown}: permissionMode {data['permissionMode']} is forbidden")
        _check_model(shown, data, policy, report)
        _check_bool_keys(shown, data, ("background",), report)
        # Only a bare name removes a tool; "Bash(curl:*)" narrows Bash but leaves it granted.
        disallowed = {
            name
            for name, raw in zip(
                tool_names(data.get("disallowedTools")),
                _raw_tools(data.get("disallowedTools")),
                strict=True,
            )
            if "(" not in raw
        }
        _check_tools(
            shown, [t for t in tool_names(data.get("tools")) if t not in disallowed], policy, report
        )
        missing_skills = sorted(set(tool_names(data.get("skills"))) - skill_names)
        if missing_skills:
            report.errors.append(f"{shown}: references unknown skills {missing_skills}")
        if "hooks" in data:
            report.checked["agent_hook_commands"] = report.checked.get(
                "agent_hook_commands", 0
            ) + check_hooks(config, data["hooks"], shown, report.errors.append)


def _check_skills(
    config: Config,
    *,
    root: Path,
    skills: dict[Path, dict[str, Any]],
    agent_names: set[str],
    policy: _ToolPolicy,
    report: AgentLayerReport,
) -> None:
    allowed = set(config.str_list("agent_layer.skill_allowed_keys"))
    required = config.str_list("agent_layer.skill_required_keys")
    for path, data in skills.items():
        shown = path.relative_to(root)
        _check_frontmatter(shown, data, allowed=allowed, required=required, report=report)
        if data.get("name") and data["name"] != path.parent.name:
            report.errors.append(f"{shown}: name {data['name']!r} != directory name")
        _check_model(shown, data, policy, report)
        _check_bool_keys(
            shown, data, ("disable-model-invocation", "user-invocable", "background"), report
        )
        _check_tools(shown, tool_names(data.get("allowed-tools")), policy, report)
        context = data.get("context")
        if context is not None and context != "fork":
            report.errors.append(f"{shown}: context must be 'fork', got {context!r}")
        agent = data.get("agent")
        if agent is not None and agent not in agent_names:
            report.errors.append(f"{shown}: agent {agent!r} is not a project or built-in agent")
        if agent is not None and context != "fork":
            report.errors.append(f"{shown}: agent is only used with context: fork")


__all__ = ["AgentLayerReport", "check_agent_layer", "tool_names"]
