from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qcal.hooks import guards
from qcal.hooks.payload import HookPayload, PayloadError
from qcal.policy import Policy


def payload(tool: str | None = "Write", **tool_input: object) -> HookPayload:
    data: dict[str, object] = {"tool_input": tool_input, "cwd": None}
    if tool is not None:
        data["tool_name"] = tool
    return HookPayload.parse(json.dumps(data))


@pytest.fixture
def policy(config) -> Policy:
    return Policy.from_config(config)


@pytest.mark.parametrize(
    ("tool", "field", "relative", "allowed", "rule"),
    [
        ("Write", "file_path", "EXPERIMENTS.yaml", False, "paths.ian_only"),
        ("Edit", "file_path", "src/qcal/handwritten/laece.py", False, "paths.ian_only"),
        ("Edit", "file_path", "runs/index.csv", False, "paths.registry_only"),
        ("Write", "file_path", ".claude/settings.json", False, "paths.enforcement_surface"),
        ("MultiEdit", "file_path", ".github/workflows/ci.yml", False, "paths.enforcement_surface"),
        ("NotebookEdit", "notebook_path", "notebooks/x.ipynb", True, "paths.ok"),
        ("NotebookEdit", "notebook_path", "notebooks/nbcu.ipynb", False, "paths.clean_room"),
        ("Write", "file_path", "src/qcal/calib/platt.py", True, "paths.ok"),
        ("Write", "path", "docs/notes.md", True, "paths.ok"),
    ],
)
def test_guard_paths(config, policy, repo: Path, tool, field, relative, allowed, rule) -> None:
    decision = guards.guard_paths(
        payload(tool, **{field: str(repo / relative)}), config=config, policy=policy, root=repo
    )
    assert (decision.allow, decision.rule) == (allowed, rule)


def test_guard_paths_denies_missing_tool_and_missing_path(config, policy, repo: Path) -> None:
    assert (
        guards.guard_paths(
            payload(None, file_path="x"), config=config, policy=policy, root=repo
        ).rule
        == "paths.no_tool"
    )
    assert (
        guards.guard_paths(payload("Write"), config=config, policy=policy, root=repo).rule
        == "paths.no_path"
    )


def test_guard_paths_ignores_non_edit_tools(config, policy, repo: Path) -> None:
    decision = guards.guard_paths(
        payload("Read", file_path=str(repo / "EXPERIMENTS.yaml")),
        config=config,
        policy=policy,
        root=repo,
    )
    assert decision.allow


def test_guard_paths_checks_every_path_field(config, policy, repo: Path) -> None:
    decision = guards.guard_paths(
        payload("Write", file_path=str(repo / "a.md"), path=str(repo / "CLAIMS.md")),
        config=config,
        policy=policy,
        root=repo,
    )
    assert decision.rule == "paths.ian_only"


def test_guard_paths_follows_symlinks(config, policy, repo: Path) -> None:
    (repo / "EXPERIMENTS.yaml").write_text("x")
    os.symlink(repo / "EXPERIMENTS.yaml", repo / "e.yaml")
    decision = guards.guard_paths(
        payload("Edit", file_path=str(repo / "e.yaml")), config=config, policy=policy, root=repo
    )
    assert decision.rule == "paths.ian_only"


def test_guard_bash_push_and_clean_room(config, policy) -> None:
    deny = guards.guard_bash(
        payload("Bash", command="git push --force origin x"),
        config=config,
        policy=policy,
        resolve_branch=lambda _c: None,
    )
    assert deny.rule == "bash.push_flag"
    clean = guards.guard_bash(
        payload("Bash", command="cp ~/NBCU/x ."), config=config, policy=policy
    )
    assert clean.rule == "bash.clean_room"
    assert guards.guard_bash(payload("Bash", command="  "), config=config, policy=policy).allow
    assert guards.guard_bash(payload("Bash"), config=config, policy=policy).allow
    ok = guards.guard_bash(
        payload("Bash", command="pytest -q"),
        config=config,
        policy=policy,
        resolve_branch=lambda _c: "claude/x",
    )
    assert ok.rule == "bash.ok"


def test_guard_bash_extra_patterns_from_config(make_config) -> None:
    config = make_config(
        '[hooks]\nextra_bash_deny = [{ pattern = "--no-verify", reason = "hooks must run" }]\n'
    )
    decision = guards.guard_bash(
        payload("Bash", command="git commit --no-verify -m x"),
        config=config,
        policy=Policy.from_config(config),
    )
    assert (decision.allow, decision.reason, decision.rule) == (
        False,
        "hooks must run",
        "bash.extra",
    )


def test_guard_bash_rejects_malformed_extra_patterns(make_config) -> None:
    config = make_config('[hooks]\nextra_bash_deny = [{ reason = "x" }]\n')
    with pytest.raises(Exception, match="pattern"):
        guards.guard_bash(
            payload("Bash", command="ls"), config=config, policy=Policy.from_config(config)
        )


def test_scope_write(config, repo: Path) -> None:
    inside = guards.scope_write(
        payload("Write", file_path=str(repo / "review/prior-art/a.md")),
        prefixes=["review/prior-art/"],
        config=config,
        root=repo,
    )
    relative = guards.scope_write(
        payload("Write", file_path="review/prior-art/b.md"),
        prefixes=["review/prior-art"],
        config=config,
        root=repo,
    )
    escape = guards.scope_write(
        payload("Write", file_path=str(repo / "review/prior-art/../../x.md")),
        prefixes=["review/prior-art/"],
        config=config,
        root=repo,
    )
    missing = guards.scope_write(payload("Write"), prefixes=["review"], config=config, root=repo)
    assert inside.allow
    assert relative.allow
    assert (escape.allow, escape.rule) == (False, "scope.outside")
    assert missing.rule == "scope.no_path"


def test_deny_read_by_category_and_glob(config, policy, repo: Path) -> None:
    by_category = guards.deny_read(
        payload("Read", file_path=str(repo / "EXPERIMENTS.yaml")),
        targets=["ian_only"],
        config=config,
        policy=policy,
        root=repo,
    )
    by_glob = guards.deny_read(
        payload("Read", file_path=str(repo / "secrets/a.txt")),
        targets=["secrets/**"],
        config=config,
        policy=policy,
        root=repo,
    )
    allowed = guards.deny_read(
        payload("Read", file_path=str(repo / "docs/a.md")),
        targets=["ian_only", "secrets/**"],
        config=config,
        policy=policy,
        root=repo,
    )
    assert by_category.rule == "read.category"
    assert by_glob.rule == "read.glob"
    assert allowed.allow


@pytest.mark.parametrize(
    ("command", "allowed", "rule"),
    [
        ("make release", True, "allow_only.ok"),
        ("make   release", True, "allow_only.ok"),
        ("make release; rm -rf /", False, "allow_only.denied"),
        ("make 'release", False, "allow_only.unparseable"),
    ],
)
def test_allow_only(command: str, allowed: bool, rule: str) -> None:
    decision = guards.allow_only(payload("Bash", command=command), commands=["make release"])
    assert (decision.allow, decision.rule) == (allowed, rule)


def test_allow_only_without_command() -> None:
    assert guards.allow_only(payload("Bash"), commands=["x"]).rule == "allow_only.missing"


def test_project_root_prefers_claude_project_dir(config, tmp_path: Path) -> None:
    assert guards.project_root(config, {"CLAUDE_PROJECT_DIR": str(tmp_path)}) == tmp_path.resolve()
    assert guards.project_root(config, {}) == config.root


@pytest.mark.parametrize("text", ["[1]", "nope", '"string"'])
def test_payload_rejects_non_objects(text: str) -> None:
    with pytest.raises(PayloadError):
        HookPayload.parse(text)


def test_payload_parsing_is_defensive() -> None:
    parsed = HookPayload.parse(
        json.dumps({"tool_name": 3, "tool_input": "x", "stop_hook_active": "yes"})
    )
    assert parsed.tool_name is None
    assert parsed.tool_input == {}
    assert parsed.stop_hook_active is False
    assert HookPayload.parse("").raw == {}
    assert HookPayload.parse('{"stop_hook_active": true}').stop_hook_active is True
    assert payload("Write", file_path="a", path="a").string_fields(["file_path", "path"]) == ["a"]


def test_worktree_inside_the_project_is_judged_relative_to_its_own_root(
    config, policy, repo: Path
) -> None:
    worktree = repo / ".claude" / "worktrees" / "agent-1"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: /elsewhere\n")  # linked worktrees have a .git file
    ok = guards.guard_paths(
        payload("Edit", file_path=str(worktree / "src/x.py")),
        config=config,
        policy=policy,
        root=repo,
    )
    blocked = guards.guard_paths(
        payload("Edit", file_path=str(worktree / "EXPERIMENTS.yaml")),
        config=config,
        policy=policy,
        root=repo,
    )
    assert ok.allow
    assert blocked.rule == "paths.ian_only"


def test_worktree_beside_the_project_is_still_protected(
    config, policy, repo: Path, tmp_path: Path
) -> None:
    worktree = tmp_path / "wt" / "claude-x"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: /elsewhere\n")
    decision = guards.guard_paths(
        payload("Write", file_path=str(worktree / "qcal.toml")),
        config=config,
        policy=policy,
        root=repo,
    )
    assert decision.rule == "paths.enforcement_surface"


def test_root_for_falls_back_to_the_project_root(repo: Path, tmp_path: Path) -> None:
    assert guards.root_for(str(tmp_path / "loose" / "file.txt"), repo) == repo
    (repo / ".git").mkdir()
    assert guards.root_for("relative/file.txt", repo) == repo


@pytest.mark.parametrize("tool_input", [{}, {"pattern": "AP"}])
def test_deny_read_fails_closed_without_a_path(config, policy, repo: Path, tool_input) -> None:
    decision = guards.deny_read(
        payload("Grep", **tool_input), targets=["ian_only"], config=config, policy=policy, root=repo
    )
    assert decision.rule == "read.no_path"


def test_deny_read_refuses_directories_that_may_hold_private_files(
    config, policy, repo: Path
) -> None:
    (repo / "docs").mkdir()
    (repo / "secrets").mkdir()

    def check(path: Path, targets: list[str]):
        return guards.deny_read(
            payload("Grep", path=str(path)),
            targets=targets,
            config=config,
            policy=policy,
            root=repo,
        )

    assert check(repo, ["secrets/**"]).rule == "read.directory"
    assert check(repo / "secrets", ["secrets/**"]).rule in {"read.glob", "read.directory"}
    assert check(repo / "docs", ["secrets/**"]).allow
    assert check(repo / "docs", ["ian_only"]).rule == "read.directory"  # **/handwritten/** anywhere
