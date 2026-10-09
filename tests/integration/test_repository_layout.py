"""The repository's own agent layer, policy and workflows are valid and consistent."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from qcal.config import load_config
from qcal.integrity.agent_layer import check_agent_layer
from qcal.integrity.licenses import read_frontmatter
from qcal.policy import Policy
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config():
    return load_config(REPO_ROOT, environ={})


def test_agent_layer_passes(config) -> None:
    report = check_agent_layer(config)
    assert report.passed, report.errors
    assert report.checked["agents"] == 2
    assert report.checked["skills"] == 1
    assert report.checked["hook_commands"] >= 4


def test_settings_wire_the_guards(config) -> None:
    settings = json.loads((REPO_ROOT / ".claude/settings.json").read_text())
    commands = {
        e: [h["command"] for g in groups for h in g["hooks"]]
        for e, groups in settings["hooks"].items()
    }
    assert any("run_hook.sh guard-paths" in c for c in commands["PreToolUse"])
    assert any("run_hook.sh guard-bash" in c for c in commands["PreToolUse"])
    assert any("--fail-open claims" in c for c in commands["Stop"])
    assert any("session_start.sh" in c for c in commands["SessionStart"])
    matchers = [g.get("matcher", "") for g in settings["hooks"]["PreToolUse"]]
    assert set(config.str_list("hooks.edit_tools")) <= set(matchers[0].split("|"))
    assert all(
        os.access(REPO_ROOT / ".claude/hooks" / n, os.X_OK)
        for n in ("run_hook.sh", "session_start.sh")
    )


def test_no_agent_has_bypass_permissions_or_write_with_network(config) -> None:
    for path in sorted((REPO_ROOT / ".claude/agents").glob("*.md")):
        meta = read_frontmatter(path.read_text())
        tools = {t.strip() for t in str(meta.get("tools", "")).split(",")}
        assert meta.get("permissionMode") != "bypassPermissions"
        assert not (
            {"Write", "Edit"} & tools
            and ({"WebFetch", "WebSearch"} & tools or meta.get("mcpServers"))
        )


def test_enforcement_surface_covers_the_agent_layer_itself(config) -> None:
    policy = Policy.from_config(config)
    for path in [
        ".claude/settings.json",
        ".claude/hooks/run_hook.sh",
        ".github/workflows/integrity.yml",
        "qcal.toml",
        "allowed_signers",
        "CLAUDE.md",
        "AGENTS.md",
        "Makefile",
        "src/qcal/hooks/guards.py",
        "src/qcal/ci/signatures.py",
        "src/qcal/resources/defaults.toml",
    ]:
        assert policy.in_categories(path, config.str_list("signing.signed_categories")), path


def test_templates_exist_for_every_init_file(config) -> None:
    templates = REPO_ROOT / "src/qcal/resources/templates"
    for key in config.str_list("init.files"):
        assert (templates / Path(config.str_value(f"paths.{key}")).name).is_file()


def test_ian_only_documents_are_not_authored_by_agents(config) -> None:
    for key in ("experiments", "decisions", "claims", "research_log"):
        assert not config.path(key).exists(), f"{key} must be created by Ian with `qcal init`"


def test_claude_md_stays_short() -> None:
    assert len((REPO_ROOT / "CLAUDE.md").read_text().splitlines()) < 200


def test_integrity_workflow_never_executes_head_code() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/integrity.yml").read_text())
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert "pull_request_target" in triggers
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["signed-and-immutable"]["steps"]
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    runs = " ".join(s.get("run", "") for s in steps)
    assert "pr/head" in runs
    assert "checkout pr/head" not in runs
    assert "git switch" not in runs


def test_ci_workflow_is_read_only_and_runs_the_suite() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    runs = " ".join(s.get("run", "") for job in workflow["jobs"].values() for s in job["steps"])
    for command in (
        "ruff check",
        "mypy",
        "pytest --cov",
        "qcal claims",
        "qcal agent-layer",
        "registry index --check",
        "registry tables --check",
        "qcal licenses",
    ):
        assert command in runs, command


def test_mcp_config_is_read_only_github() -> None:
    servers = json.loads((REPO_ROOT / ".mcp.json").read_text())["mcpServers"]
    assert set(servers) == {"github"}
    assert servers["github"]["headers"]["X-MCP-Readonly"] == "true"
    assert "${" in servers["github"]["headers"]["Authorization"]


def test_repository_policy_overrides(config) -> None:
    assert "civ" in config.str_list("git.protected_branches")
    assert config.str_value("signing.mode") in {"bootstrap", "enforce"}
