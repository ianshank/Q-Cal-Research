"""Agent-layer tool policy, model allowlist, cross-references and hook-settings validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qcal.config import Config
from qcal.integrity.agent_layer import check_agent_layer, tool_names
from tests.conftest import write

AGENT = ".claude/agents/reviewer.md"
SKILL = ".claude/skills/weekly/SKILL.md"
SETTINGS = ".claude/settings.json"
BASE_AGENT = "name: reviewer\ndescription: Reviews things.\n"  # tools added per test
AGENT_WITH_TOOLS = BASE_AGENT + "tools: Read\n"
BASE_SKILL = "name: weekly\ndescription: Weekly review.\n"


def front(root: Path, relative: str, text: str) -> Path:
    return write(root, relative, f"---\n{text}---\nBody.\n")


def settings(root: Path, hooks: dict[str, list[dict[str, Any]]]) -> Path:
    return write(root, SETTINGS, json.dumps({"hooks": hooks}))


def group(matcher: Any) -> dict[str, Any]:
    return {"matcher": matcher, "hooks": [{"type": "prompt", "prompt": "x"}]}


# --- tool names -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("Read, Grep", ["Read", "Grep"]),
        (["Read", "Bash(git log:*)"], ["Read", "Bash"]),
        ("Bash(make check), mcp__github__get_me", ["Bash", "mcp__github__get_me"]),
        (" , ,", []),
    ],
)
def test_tool_names(value: Any, expected: list[str]) -> None:
    assert tool_names(value) == expected


# --- tool policy ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tools",
    ["Read, Write, WebFetch", "Bash, WebSearch", "Edit, mcp__github__create_issue"],
    ids=["write+fetch", "bash+search", "edit+mcp"],
)
def test_agent_combining_write_and_network_tools_fails(
    config: Config, repo: Path, tools: str
) -> None:
    front(repo, AGENT, BASE_AGENT + f"tools: {tools}\n")

    errors = check_agent_layer(config).errors

    assert len(errors) == 1
    assert "combines write-capable tools" in errors[0]


def test_disallowed_tools_remove_the_conflict(config: Config, repo: Path) -> None:
    front(repo, AGENT, BASE_AGENT + "tools: Read, Bash, WebFetch\ndisallowedTools: Bash\n")

    assert check_agent_layer(config).errors == []


def test_read_only_network_agent_passes(config: Config, repo: Path) -> None:
    front(repo, AGENT, BASE_AGENT + "tools: Read, Grep, WebFetch, WebSearch\n")

    assert check_agent_layer(config).errors == []


def test_skill_allowed_tools_follow_the_same_policy(config: Config, repo: Path) -> None:
    front(repo, SKILL, BASE_SKILL + "allowed-tools: Write, WebFetch\n")

    assert any("combines write-capable" in e for e in check_agent_layer(config).errors)


def test_tool_policy_lists_are_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config("[agent_layer]\nnetwork_tools = []\nnetwork_tool_prefixes = []\n")
    front(repo, AGENT, BASE_AGENT + "tools: Write, WebFetch\n")

    assert check_agent_layer(config).errors == []


# --- models ---------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["inherit", "opus", "sonnet", "haiku", "claude-sonnet-5-5"])
def test_allowed_models_pass(config: Config, repo: Path, model: str) -> None:
    front(repo, AGENT, AGENT_WITH_TOOLS + f"model: {model}\n")

    assert check_agent_layer(config).errors == []


@pytest.mark.parametrize("model", ["gpt-4", "Opus", "claude", "1"])
def test_unknown_models_fail(config: Config, repo: Path, model: str) -> None:
    front(repo, SKILL, BASE_SKILL + f"model: {model}\n")

    errors = check_agent_layer(config).errors

    assert len(errors) == 1
    assert "is not in the allowlist" in errors[0]


# --- typed keys and references ----------------------------------------------------------------


@pytest.mark.parametrize("key", ["disable-model-invocation", "user-invocable", "background"])
def test_skill_boolean_keys_must_be_booleans(config: Config, repo: Path, key: str) -> None:
    front(repo, SKILL, BASE_SKILL + f'{key}: "yes"\n')

    assert check_agent_layer(config).errors == [f"{SKILL}: {key} must be true or false, got 'yes'"]


def test_skill_name_must_match_its_directory(config: Config, repo: Path) -> None:
    front(repo, SKILL, "name: other\ndescription: d\n")

    assert check_agent_layer(config).errors == [f"{SKILL}: name 'other' != directory name"]


@pytest.mark.parametrize("agent", ["reviewer", "Explore", "general-purpose", "Plan"])
def test_fork_skill_naming_a_real_agent_passes(config: Config, repo: Path, agent: str) -> None:
    front(repo, AGENT, AGENT_WITH_TOOLS)
    front(repo, SKILL, BASE_SKILL + f"context: fork\nagent: {agent}\n")

    assert check_agent_layer(config).errors == []


def test_fork_skill_naming_a_missing_agent_fails(config: Config, repo: Path) -> None:
    front(repo, SKILL, BASE_SKILL + "context: fork\nagent: ghost\n")

    assert check_agent_layer(config).errors == [
        f"{SKILL}: agent 'ghost' is not a project or built-in agent"
    ]


def test_agent_without_fork_context_is_flagged(config: Config, repo: Path) -> None:
    front(repo, SKILL, BASE_SKILL + "agent: Explore\n")

    assert check_agent_layer(config).errors == [f"{SKILL}: agent is only used with context: fork"]


def test_unknown_context_value_fails(config: Config, repo: Path) -> None:
    front(repo, SKILL, BASE_SKILL + "context: spawn\n")

    assert check_agent_layer(config).errors == [f"{SKILL}: context must be 'fork', got 'spawn'"]


def test_agent_skills_must_exist(config: Config, repo: Path) -> None:
    front(repo, SKILL, BASE_SKILL)
    front(repo, AGENT, AGENT_WITH_TOOLS + "skills: [weekly, missing]\n")

    assert check_agent_layer(config).errors == [f"{AGENT}: references unknown skills ['missing']"]


def test_skill_required_keys_are_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config('[agent_layer]\nskill_required_keys = ["description", "argument-hint"]\n')
    front(repo, SKILL, BASE_SKILL)

    assert check_agent_layer(config).errors == [f"{SKILL}: missing required keys ['argument-hint']"]


# --- hook settings --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "matcher"),
    [
        ("PreToolUse", "Edit|Write|MultiEdit|NotebookEdit"),
        ("PreToolUse", "Bash"),
        ("PostToolUse", "mcp__github__get_me"),
        ("PreToolUse", "mcp__.*"),
        ("PreToolUse", "*"),
        ("PreToolUse", ""),
        ("Stop", None),
        ("SessionStart", "startup|resume"),
        ("UserPromptSubmit", None),
    ],
)
def test_valid_hook_events_and_matchers_pass(
    config: Config, repo: Path, event: str, matcher: str | None
) -> None:
    settings(repo, {event: [group(matcher)]})

    assert check_agent_layer(config).errors == []


@pytest.mark.parametrize(
    ("event", "matcher", "fragment"),
    [
        ("PreToolUse", "Edit|Wrte", "unknown tools ['Wrte']"),
        ("PreToolUse", "Bash(", "is not a valid regex"),
        ("PreToolUse", 3, "matcher must be a string"),
        ("SessionStart", "startup|boot", "unknown sources ['boot']"),
        ("PreToolUze", None, "unknown hook event 'PreToolUze'"),
    ],
)
def test_invalid_hook_settings_fail(
    config: Config, repo: Path, event: str, matcher: Any, fragment: str
) -> None:
    settings(repo, {event: [group(matcher)]})

    errors = check_agent_layer(config).errors

    assert len(errors) == 1
    assert fragment in errors[0]


def test_hook_event_groups_must_be_a_list(config: Config, repo: Path) -> None:
    settings(repo, {"Stop": {"hooks": []}})  # type: ignore[dict-item]

    assert check_agent_layer(config).errors == [
        f"{SETTINGS}: Stop must be a list of matcher groups"
    ]


@pytest.mark.parametrize("payload", ["[]", '"x"', "3"])
def test_settings_top_level_must_be_an_object(config: Config, repo: Path, payload: str) -> None:
    write(repo, SETTINGS, payload)

    assert check_agent_layer(config).errors == [f"{SETTINGS}: top level must be a JSON object"]


def test_known_tools_are_configurable(make_config: Callable[[str], Config], repo: Path) -> None:
    config = make_config('[agent_layer]\nknown_tools = ["Bash", "CustomTool"]\n')
    settings(repo, {"PreToolUse": [group("CustomTool|Bash")]})

    assert check_agent_layer(config).errors == []


# --- wrapped qcal hooks and guard matchers ----------------------------------------------------

WRAPPER = '"$CLAUDE_PROJECT_DIR"/.claude/hooks/run_hook.sh'


def wrapper_settings(root: Path, matcher: str, *commands: str) -> None:
    script_path = root / ".claude/hooks/run_hook.sh"
    write(root, ".claude/hooks/run_hook.sh", "#!/bin/sh\n").chmod(0o755)
    assert script_path.is_file()
    hooks = [{"type": "command", "command": f"{WRAPPER} {c}"} for c in commands]
    settings(root, {"PreToolUse": [{"matcher": matcher, "hooks": hooks}]})


def test_wrapped_hook_names_must_exist(config: Config, repo: Path) -> None:
    wrapper_settings(repo, "Bash", "guard-bsh")

    assert check_agent_layer(config).errors == [
        f"{SETTINGS}: PreToolUse hook '{WRAPPER} guard-bsh' runs unknown qcal hook 'guard-bsh'"
    ]


def test_fail_open_flag_is_skipped_when_reading_the_hook_name(config: Config, repo: Path) -> None:
    wrapper_settings(repo, "Bash", "--fail-open guard-bash")

    assert check_agent_layer(config).errors == []


@pytest.mark.parametrize(
    ("matcher", "hook", "missing"),
    [
        ("Edit|Write", "guard-paths", "['MultiEdit', 'NotebookEdit']"),
        ("Edit|Write|MultiEdit|NotebookEdit", "guard-bash", "['Bash']"),
    ],
)
def test_guard_matchers_must_cover_their_tools(
    config: Config, repo: Path, matcher: str, hook: str, missing: str
) -> None:
    wrapper_settings(repo, matcher, hook)

    assert check_agent_layer(config).errors == [
        f"{SETTINGS}: PreToolUse matcher {matcher!r} for {hook} misses tools {missing}"
    ]


def test_wildcard_matcher_covers_every_tool(config: Config, repo: Path) -> None:
    wrapper_settings(repo, "*", "guard-paths", "guard-bash")

    assert check_agent_layer(config).errors == []


def test_matcher_groups_must_be_mappings(config: Config, repo: Path) -> None:
    settings(repo, {"Stop": ["not a group"]})  # type: ignore[list-item]

    assert check_agent_layer(config).errors == [f"{SETTINGS}: Stop matcher groups must be mappings"]


def test_agent_frontmatter_hooks_are_validated(config: Config, repo: Path) -> None:
    front(repo, AGENT, AGENT_WITH_TOOLS + "hooks:\n  PreToolUze: []\n")
    front(repo, ".claude/agents/other.md", "name: other\ndescription: d\nhooks: [1]\n")

    errors = check_agent_layer(config).errors

    assert f"{AGENT}: unknown hook event 'PreToolUze' (Claude Code would ignore it)" in errors
    assert (
        ".claude/agents/other.md: hooks must be a mapping of event names to matcher groups"
        in errors
    )


def test_agents_must_declare_their_tools(config: Config, repo: Path) -> None:
    front(repo, AGENT, BASE_AGENT)

    assert check_agent_layer(config).errors == [f"{AGENT}: missing required keys ['tools']"]
