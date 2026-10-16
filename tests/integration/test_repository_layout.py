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

pytestmark = [pytest.mark.integration, pytest.mark.rule("A3")]


@pytest.fixture(scope="module")
def config():
    return load_config(REPO_ROOT, environ={})


def test_agent_layer_passes(config) -> None:
    report = check_agent_layer(config)
    assert report.passed, report.errors
    agents = list(REPO_ROOT.glob(config.str_value("agent_layer.agents_glob")))
    skills = list(REPO_ROOT.glob(config.str_value("agent_layer.skills_glob")))
    assert report.checked["agents"] == len(agents) >= 2
    assert report.checked["skills"] == len(skills) >= 1
    assert report.checked["hook_commands"] >= 4
    assert report.checked["command_refs"] > 0


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


def test_user_invoked_skills_cannot_be_triggered_by_the_model(config) -> None:
    for name in ("pre-pr", "change-proposal"):
        meta = read_frontmatter((REPO_ROOT / f".claude/skills/{name}/SKILL.md").read_text())
        assert meta["disable-model-invocation"] is True, name


def test_leakage_checker_bash_is_limited_to_the_leakage_command(config) -> None:
    meta = read_frontmatter((REPO_ROOT / ".claude/agents/data-leakage-checker.md").read_text())
    (group,) = meta["hooks"]["PreToolUse"]
    assert group["matcher"] == "Bash"
    assert "allow-only" in group["hooks"][0]["command"]


def test_infrastructure_files_are_present_and_protected(config) -> None:
    policy = Policy.from_config(config)
    categories = config.str_list("signing.signed_categories")
    for path in (".gitleaks.toml", ".github/dependabot.yml", ".github/workflows/ci.yml"):
        assert (REPO_ROOT / path).is_file(), path
        assert policy.in_categories(path, categories), path
    assert (REPO_ROOT / "Dockerfile").is_file()
    assert ".git" in (REPO_ROOT / ".dockerignore").read_text().splitlines()
    assert os.access(REPO_ROOT / "scripts/nightly.sh", os.X_OK)


def test_ci_runs_every_suite_and_the_new_jobs() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    assert {"quality", "tests", "integrity", "secrets", "container"} <= set(workflow["jobs"])
    runs = " ".join(s.get("run", "") for job in workflow["jobs"].values() for s in job["steps"])
    assert "--junitxml" in runs
    assert "sha256sum -c" in runs  # the gitleaks binary is checksum-verified


def test_integrity_workflow_runs_the_review_check_without_interpolating_the_branch() -> None:
    text = (REPO_ROOT / ".github/workflows/integrity.yml").read_text()
    assert "qcal ci review-check" in text
    assert '--branch "$HEAD_REF"' in text
    assert "${{ github.event.pull_request.head.ref }}" not in text.split("steps:", 1)[1]


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


def test_mcp_config_is_read_only() -> None:
    servers = json.loads((REPO_ROOT / ".mcp.json").read_text())["mcpServers"]
    assert set(servers) == {"github", "huggingface"}
    assert servers["github"]["headers"]["X-MCP-Readonly"] == "true"
    assert "${" in servers["github"]["headers"]["Authorization"]
    # prior-art-scout's hub access is anonymous: no token, so nothing can be published with it.
    assert servers["huggingface"] == {"type": "http", "url": "https://huggingface.co/mcp"}


def test_repository_policy_overrides(config) -> None:
    assert "civ" in config.str_list("git.protected_branches")
    assert config.str_value("signing.mode") in {"bootstrap", "enforce"}


# --- coverage exclusions never hide a function body ------------------------------------------


def _exclusions() -> list[str]:
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    return list(data["tool"]["coverage"]["report"]["exclude_also"])


def _is_stub(body: list) -> bool:
    import ast

    rest = (
        body[1:]
        if body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
        else body
    )
    return all(
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is ...
        for s in rest
    )


def test_no_coverage_exclusion_matches_a_signature_line() -> None:
    """coverage.py drops a whole function when any line from its first decorator to its colon
    matches an exclusion; an unanchored pattern once hid 30 functions that way."""
    import ast
    import re

    patterns = [re.compile(p) for p in _exclusions()]
    hidden: list[str] = []
    for package in ("src/qcal", "src/qcal_lab"):
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            source = path.read_text("utf-8").splitlines()
            for node in ast.walk(ast.parse("\n".join(source))):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    continue
                if _is_stub(node.body):
                    continue
                first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                for number in range(first, node.body[0].lineno):
                    line = source[number - 1]
                    if any(p.search(line) for p in patterns):
                        relative = path.relative_to(REPO_ROOT)
                        hidden.append(f"{relative}:{number}: {line.strip()}")
    assert hidden == []


def test_the_exclusion_patterns_still_exclude_what_they_are_for() -> None:
    import re

    patterns = [re.compile(p) for p in _exclusions()]

    def excluded(line: str) -> bool:
        return any(p.search(line) for p in patterns)

    assert excluded("    ...")
    assert excluded("    def predict(self, ids: Sequence[str]) -> Sequence[X]: ...")
    assert excluded("if TYPE_CHECKING:")
    assert excluded("        raise NotImplementedError")
    assert not excluded("def f(x: int) -> tuple[str, ...]:")
    assert not excluded("    key: tuple[str, ...],")
    assert not excluded('    out.write("...")')


# --- the workflows' qcal commands exist ------------------------------------------------------


def _workflow_commands() -> list[tuple[str, list[str]]]:
    import shlex

    found: list[tuple[str, list[str]]] = []
    for path in sorted((REPO_ROOT / ".github/workflows").glob("*.yml")):
        for job in yaml.safe_load(path.read_text("utf-8"))["jobs"].values():
            for step in job.get("steps", []):
                for line in str(step.get("run", "")).splitlines():
                    if "-m qcal " not in line:
                        continue
                    words = shlex.split(line)
                    if "qcal" in words and "-m" in words and words[words.index("qcal") - 1] == "-m":
                        argv = words[words.index("qcal") + 1 :]
                        found.append((path.name, [w.replace("$", "") for w in argv]))
    return found


def test_every_qcal_command_in_a_workflow_parses() -> None:
    """integrity.yml runs the base branch's qcal: a flag removed or made exclusive here would
    break the trusted job only after merge."""
    from qcal.cli.app import build_parser

    commands = _workflow_commands()
    assert {name for name, _ in commands} == {"ci.yml", "integrity.yml"}
    parser = build_parser()
    for name, argv in commands:
        try:
            parser.parse_args(argv)
        except SystemExit:  # pragma: no cover - the assertion below names the command
            pytest.fail(f"{name}: qcal {' '.join(argv)} does not parse")
