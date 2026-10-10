"""Agent-layer validation: subagent/skill frontmatter, hook scripts and MCP servers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qcal.config import Config
from qcal.integrity.agent_layer import AgentLayerReport, check_agent_layer
from tests.conftest import write

AGENT = ".claude/agents/reviewer.md"
SKILL = ".claude/skills/weekly/SKILL.md"
SETTINGS = ".claude/settings.json"
MCP = ".mcp.json"
VALID_AGENT = "name: reviewer\ndescription: Reviews things.\ntools: Read, Grep\n"
VALID_SKILL = "name: weekly\ndescription: Weekly review.\nargument-hint: <week>\n"


def frontmatter_file(root: Path, relative: str, front: str | None) -> Path:
    text = "Body.\n" if front is None else f"---\n{front}---\nBody.\n"
    return write(root, relative, text)


def hooks_settings(root: Path, hooks: dict[str, list[dict[str, Any]]]) -> Path:
    data = {"hooks": {event: [{"matcher": "*", "hooks": items}] for event, items in hooks.items()}}
    return write(root, SETTINGS, json.dumps(data))


def command_hook(command: str) -> dict[str, Any]:
    return {"type": "command", "command": command}


def script(root: Path, relative: str, *, executable: bool = True) -> Path:
    path = write(root, relative, "#!/bin/sh\nexit 0\n")
    path.chmod(0o755 if executable else 0o644)
    return path


def mcp(root: Path, servers: dict[str, Any]) -> Path:
    return write(root, MCP, json.dumps({"mcpServers": servers}))


# --- empty project ------------------------------------------------------------------------


def test_project_without_agent_layer_passes_and_checks_nothing(config: Config) -> None:
    report = check_agent_layer(config)

    assert report.passed
    assert report.checked == {
        "agents": 0,
        "skills": 0,
        "hook_commands": 0,
        "mcp_servers": 0,
        "command_refs": 0,
    }


def test_report_dict(config: Config, repo: Path) -> None:
    frontmatter_file(repo, AGENT, None)

    data = check_agent_layer(config).to_dict()

    assert data["verdict"] == "FAIL"
    assert data["errors"] == [f"{AGENT}: missing or empty YAML frontmatter"]
    assert data["checked"]["agents"] == 1


def test_empty_report_passes() -> None:
    assert AgentLayerReport().to_dict() == {"verdict": "PASS", "errors": [], "checked": {}}


# --- subagents ----------------------------------------------------------------------------


def test_valid_agent_passes(config: Config, repo: Path) -> None:
    frontmatter_file(repo, AGENT, VALID_AGENT)

    report = check_agent_layer(config)

    assert report.passed
    assert report.checked["agents"] == 1


def test_every_documented_agent_key_is_accepted(config: Config, repo: Path) -> None:
    keys = config.str_list("agent_layer.agent_allowed_keys")
    typed = {"model": "sonnet", "background": "false", "skills": "[]", "hooks": "{}"}
    extra = "".join(f"{k}: {typed.get(k, 'x')}\n" for k in keys if k not in {"name", "description"})
    frontmatter_file(repo, AGENT, VALID_AGENT.replace("tools: Read, Grep\n", "") + extra)

    assert check_agent_layer(config).errors == []


@pytest.mark.parametrize(
    "front", [None, "", "# just a comment\n"], ids=["no-fence", "empty", "comment-only"]
)
def test_agent_without_frontmatter_is_an_error(
    config: Config, repo: Path, front: str | None
) -> None:
    frontmatter_file(repo, AGENT, front)

    assert check_agent_layer(config).errors == [f"{AGENT}: missing or empty YAML frontmatter"]


def test_agent_with_malformed_yaml_is_an_error(config: Config, repo: Path) -> None:
    frontmatter_file(repo, AGENT, "name: reviewer\ntools: [Read\n")

    errors = check_agent_layer(config).errors

    assert len(errors) == 1
    assert errors[0].startswith(f"{AGENT}: invalid YAML frontmatter:")


def test_unknown_agent_keys_are_listed_sorted(config: Config, repo: Path) -> None:
    frontmatter_file(repo, AGENT, VALID_AGENT + "tool: Read\npermisionMode: plan\n")

    assert check_agent_layer(config).errors == [
        (
            f"{AGENT}: unknown frontmatter keys ['permisionMode', 'tool'] "
            "(Claude Code would ignore them)"
        )
    ]


@pytest.mark.parametrize(
    ("front", "missing"),
    [
        ("name: reviewer\n", ["description"]),
        ("name: reviewer\ndescription: ''\n", ["description"]),
        ("description: d\n", ["name"]),
        ("tools: Read\n", ["name", "description"]),
    ],
)
def test_missing_or_empty_required_agent_keys_are_errors(
    config: Config, repo: Path, front: str, missing: list[str]
) -> None:
    frontmatter_file(repo, AGENT, front)

    assert check_agent_layer(config).errors == [f"{AGENT}: missing required keys {missing}"]


def test_agent_name_must_equal_file_stem(config: Config, repo: Path) -> None:
    frontmatter_file(repo, AGENT, "name: other\ndescription: d\n")

    assert check_agent_layer(config).errors == [f"{AGENT}: name 'other' != file name"]


def test_forbidden_permission_mode_is_an_error(config: Config, repo: Path) -> None:
    frontmatter_file(repo, AGENT, VALID_AGENT + "permissionMode: bypassPermissions\n")

    assert check_agent_layer(config).errors == [
        f"{AGENT}: permissionMode bypassPermissions is forbidden"
    ]


@pytest.mark.parametrize("mode", ["default", "acceptEdits", "plan"])
def test_other_permission_modes_are_allowed(config: Config, repo: Path, mode: str) -> None:
    frontmatter_file(repo, AGENT, VALID_AGENT + f"permissionMode: {mode}\n")

    assert check_agent_layer(config).passed


def test_forbidden_permission_modes_are_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config('[agent_layer]\nforbidden_permission_modes = ["acceptEdits"]\n')
    frontmatter_file(repo, AGENT, VALID_AGENT + "permissionMode: acceptEdits\n")
    frontmatter_file(repo, ".claude/agents/other.md", "name: other\ndescription: d\n")

    assert check_agent_layer(config).errors == [f"{AGENT}: permissionMode acceptEdits is forbidden"]


def test_one_file_can_report_several_problems(config: Config, repo: Path) -> None:
    frontmatter_file(repo, AGENT, "name: other\nbogus: 1\npermissionMode: bypassPermissions\n")

    assert check_agent_layer(config).errors == [
        f"{AGENT}: unknown frontmatter keys ['bogus'] (Claude Code would ignore them)",
        f"{AGENT}: missing required keys ['description']",
        f"{AGENT}: name 'other' != file name",
        f"{AGENT}: permissionMode bypassPermissions is forbidden",
    ]


def test_files_outside_the_agent_glob_are_ignored(config: Config, repo: Path) -> None:
    write(repo, ".claude/agents/README.txt", "notes\n")
    write(repo, ".claude/agents/drafts/half.md", "no frontmatter\n")
    write(repo, "agents/loose.md", "no frontmatter\n")

    report = check_agent_layer(config)

    assert report.passed
    assert report.checked["agents"] == 0


# --- skills -------------------------------------------------------------------------------


def test_valid_skill_passes(config: Config, repo: Path) -> None:
    frontmatter_file(repo, SKILL, VALID_SKILL)

    report = check_agent_layer(config)

    assert report.passed
    assert report.checked["skills"] == 1


def test_skill_needs_only_a_description(config: Config, repo: Path) -> None:
    frontmatter_file(repo, SKILL, "description: Weekly review.\n")

    assert check_agent_layer(config).passed


@pytest.mark.parametrize(
    ("front", "error"),
    [
        (None, f"{SKILL}: missing or empty YAML frontmatter"),
        ("name: weekly\n", f"{SKILL}: missing required keys ['description']"),
        (
            "description: d\ntools: Read\n",
            f"{SKILL}: unknown frontmatter keys ['tools'] (Claude Code would ignore them)",
        ),
    ],
    ids=["no-frontmatter", "no-description", "agent-only-key"],
)
def test_invalid_skill_frontmatter_is_an_error(
    config: Config, repo: Path, front: str | None, error: str
) -> None:
    frontmatter_file(repo, SKILL, front)

    assert check_agent_layer(config).errors == [error]


def test_skill_with_malformed_yaml_is_an_error(config: Config, repo: Path) -> None:
    frontmatter_file(repo, SKILL, "description: [unclosed\n")

    errors = check_agent_layer(config).errors

    assert len(errors) == 1
    assert errors[0].startswith(f"{SKILL}: invalid YAML frontmatter:")


def test_only_skill_md_one_level_down_is_a_skill(config: Config, repo: Path) -> None:
    write(repo, ".claude/skills/SKILL.md", "no frontmatter\n")
    write(repo, ".claude/skills/a/b/SKILL.md", "no frontmatter\n")
    write(repo, ".claude/skills/a/notes.md", "no frontmatter\n")

    report = check_agent_layer(config)

    assert report.passed
    assert report.checked["skills"] == 0


def test_agent_and_skill_globs_are_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config(
        '[agent_layer]\nagents_glob = "agents/*.md"\nskills_glob = "skills/*.md"\n'
    )
    frontmatter_file(repo, AGENT, None)
    frontmatter_file(repo, "agents/x.md", None)
    frontmatter_file(repo, "skills/y.md", "description: d\n")

    report = check_agent_layer(config)

    assert report.errors == ["agents/x.md: missing or empty YAML frontmatter"]
    assert (report.checked["agents"], report.checked["skills"]) == (1, 1)


# --- hook commands ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        '"$CLAUDE_PROJECT_DIR"/.claude/hooks/guard.sh',
        "$CLAUDE_PROJECT_DIR/.claude/hooks/guard.sh",
        '"${CLAUDE_PROJECT_DIR}/.claude/hooks/guard.sh" pre-tool',
        'bash "$CLAUDE_PROJECT_DIR/.claude/hooks/guard.sh" --flag',
    ],
)
def test_hook_script_located_via_project_dir_passes(
    config: Config, repo: Path, command: str
) -> None:
    script(repo, ".claude/hooks/guard.sh")
    hooks_settings(repo, {"PreToolUse": [command_hook(command)]})

    report = check_agent_layer(config)

    assert report.errors == []
    assert report.checked["hook_commands"] == 1


@pytest.mark.parametrize(
    "command", [".claude/hooks/guard.sh", "/usr/bin/true", "echo hello", "$HOME/guard.sh"]
)
def test_hook_not_located_via_project_dir_is_an_error(
    config: Config, repo: Path, command: str
) -> None:
    script(repo, ".claude/hooks/guard.sh")
    hooks_settings(repo, {"Stop": [command_hook(command)]})

    assert check_agent_layer(config).errors == [
        f"{SETTINGS}: Stop hook {command!r} should locate its script via $CLAUDE_PROJECT_DIR"
    ]


def test_missing_hook_script_is_an_error(config: Config, repo: Path) -> None:
    hooks_settings(repo, {"PreToolUse": [command_hook('"$CLAUDE_PROJECT_DIR"/hooks/gone.sh')]})

    assert check_agent_layer(config).errors == [
        f"{SETTINGS}: PreToolUse hook script hooks/gone.sh is missing"
    ]


def test_hook_script_that_is_a_directory_is_missing(config: Config, repo: Path) -> None:
    (repo / "hooks" / "dir.sh").mkdir(parents=True)
    hooks_settings(repo, {"PreToolUse": [command_hook("$CLAUDE_PROJECT_DIR/hooks/dir.sh")]})

    assert check_agent_layer(config).errors == [
        f"{SETTINGS}: PreToolUse hook script hooks/dir.sh is missing"
    ]


def test_non_executable_hook_script_is_an_error(config: Config, repo: Path) -> None:
    script(repo, ".claude/hooks/guard.sh", executable=False)
    hooks_settings(
        repo, {"PreToolUse": [command_hook("$CLAUDE_PROJECT_DIR/.claude/hooks/guard.sh")]}
    )

    assert check_agent_layer(config).errors == [
        f"{SETTINGS}: PreToolUse hook script .claude/hooks/guard.sh is not executable"
    ]


def test_hook_commands_are_counted_across_events_and_groups(config: Config, repo: Path) -> None:
    script(repo, ".claude/hooks/guard.sh")
    ok = command_hook("$CLAUDE_PROJECT_DIR/.claude/hooks/guard.sh")
    data = {
        "hooks": {
            "PreToolUse": [
                {"matcher": "Edit", "hooks": [ok, ok]},
                {"matcher": "Bash", "hooks": [ok]},
            ],
            "Stop": [{"hooks": [ok]}],
            "SessionStart": [{"matcher": "startup"}],
        }
    }
    write(repo, SETTINGS, json.dumps(data))

    report = check_agent_layer(config)

    assert report.passed
    assert report.checked["hook_commands"] == 4


@pytest.mark.parametrize("data", [{}, {"hooks": None}, {"hooks": {}}, {"permissions": {}}])
def test_settings_without_hooks_count_zero(config: Config, repo: Path, data: Any) -> None:
    write(repo, SETTINGS, json.dumps(data))

    report = check_agent_layer(config)

    assert report.passed
    assert report.checked["hook_commands"] == 0


def test_invalid_settings_json_is_an_error(config: Config, repo: Path) -> None:
    write(repo, SETTINGS, "{not json")

    report = check_agent_layer(config)

    assert len(report.errors) == 1
    assert report.errors[0].startswith(f"{SETTINGS}: invalid JSON (")
    assert report.checked["hook_commands"] == 0


def test_settings_file_location_is_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config('[agent_layer]\nsettings_file = "conf/settings.json"\n')
    write(
        repo,
        "conf/settings.json",
        json.dumps({"hooks": {"Stop": [{"hooks": [command_hook("x")]}]}}),
    )

    assert check_agent_layer(config).errors == [
        "conf/settings.json: Stop hook 'x' should locate its script via $CLAUDE_PROJECT_DIR"
    ]


@pytest.mark.parametrize(
    "hook",
    [
        {"type": "prompt", "prompt": "Check the claims are tagged."},
        {"type": "agent", "prompt": "Review the diff."},
        {"type": "http", "url": "https://example.invalid/hook"},
    ],
    ids=["prompt", "agent", "http"],
)
def test_hooks_without_a_script_are_skipped(config: Config, repo: Path, hook: Any) -> None:
    hooks_settings(repo, {"Stop": [hook]})

    report = check_agent_layer(config)

    assert report.errors == []
    assert report.checked["hook_commands"] == 0


def test_hook_without_a_type_is_treated_as_a_command(config: Config, repo: Path) -> None:
    hooks_settings(repo, {"Stop": [{"command": "./guard.sh"}]})

    assert check_agent_layer(config).errors == [
        f"{SETTINGS}: Stop hook './guard.sh' should locate its script via $CLAUDE_PROJECT_DIR"
    ]


def test_command_hooks_beside_prompt_hooks_are_still_checked(config: Config, repo: Path) -> None:
    hooks_settings(
        repo,
        {"Stop": [{"type": "prompt", "prompt": "p"}, command_hook("$CLAUDE_PROJECT_DIR/gone.sh")]},
    )

    report = check_agent_layer(config)

    assert report.errors == [f"{SETTINGS}: Stop hook script gone.sh is missing"]
    assert report.checked["hook_commands"] == 1


# --- MCP servers --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server",
    [
        {"type": "http", "url": "https://example.invalid/mcp"},
        {"type": "sse", "url": "https://example.invalid/sse"},
        {"command": "npx", "args": ["server"]},
        {
            "type": "http",
            "url": "https://x.invalid",
            "headers": {"Authorization": "Bearer ${TOKEN}"},
        },
        {"type": "http", "url": "https://x.invalid", "headers": {"X-Readonly": "true"}},
    ],
)
def test_well_formed_mcp_server_passes(config: Config, repo: Path, server: Any) -> None:
    mcp(repo, {"svc": server})

    report = check_agent_layer(config)

    assert report.passed
    assert report.checked["mcp_servers"] == 1


def test_remote_mcp_server_without_type_is_an_error(config: Config, repo: Path) -> None:
    mcp(repo, {"docs": {"url": "https://example.invalid/mcp"}})

    assert check_agent_layer(config).errors == [
        f"{MCP}: remote server 'docs' needs an explicit 'type'"
    ]


def test_literal_bearer_token_is_an_error(config: Config, repo: Path) -> None:
    mcp(
        repo,
        {
            "gh": {
                "type": "http",
                "url": "https://x.invalid",
                "headers": {"Authorization": "Bearer abc123"},
            }
        },
    )

    assert check_agent_layer(config).errors == [f"{MCP}: server 'gh' embeds a literal credential"]


def test_every_mcp_server_is_checked(config: Config, repo: Path) -> None:
    mcp(
        repo,
        {
            "a": {"url": "https://a.invalid"},
            "b": {"type": "http", "url": "https://b.invalid"},
            "c": {"type": "http", "url": "https://c.invalid", "headers": {"Auth": "Bearer t"}},
        },
    )

    report = check_agent_layer(config)

    assert report.errors == [
        f"{MCP}: remote server 'a' needs an explicit 'type'",
        f"{MCP}: server 'c' embeds a literal credential",
    ]
    assert report.checked["mcp_servers"] == 3


def test_mcp_file_without_servers_counts_zero(config: Config, repo: Path) -> None:
    write(repo, MCP, "{}")

    report = check_agent_layer(config)

    assert report.passed
    assert report.checked["mcp_servers"] == 0


def test_invalid_mcp_json_is_an_error(config: Config, repo: Path) -> None:
    write(repo, MCP, "[1, 2")

    report = check_agent_layer(config)

    assert len(report.errors) == 1
    assert report.errors[0].startswith(f"{MCP}: invalid JSON (")


def test_mcp_file_location_is_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config('[agent_layer]\nmcp_file = "conf/mcp.json"\n')
    write(repo, "conf/mcp.json", json.dumps({"mcpServers": {"r": {"url": "https://r.invalid"}}}))

    assert check_agent_layer(config).errors == [
        "conf/mcp.json: remote server 'r' needs an explicit 'type'"
    ]


@pytest.mark.parametrize("value", ["bearer abc", "BEARER abc", "BeArEr abc"])
def test_literal_bearer_token_is_detected_whatever_its_case(
    config: Config, repo: Path, value: str
) -> None:
    mcp(
        repo,
        {"gh": {"type": "http", "url": "https://x.invalid", "headers": {"Authorization": value}}},
    )

    assert check_agent_layer(config).errors == [f"{MCP}: server 'gh' embeds a literal credential"]
