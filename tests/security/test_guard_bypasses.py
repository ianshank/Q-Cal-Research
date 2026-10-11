"""Bypass attempts against the hooks, from the peer review (findings 7, 8 and 9)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from qcal.hooks.bash import analyze
from qcal.hooks.guards import root_for
from tests.conftest import REPO_ROOT, run_git, write

pytestmark = pytest.mark.rule("C0")

WRAPPER = REPO_ROOT / ".claude" / "hooks" / "run_hook.sh"
PROTECTED = ["main", "civ"]
DENY_FLAGS = ["--force", "-f", "--force-with-lease", "--mirror", "--all", "--delete", "-d"]


def _payload(tool: str, **tool_input: str) -> str:
    return json.dumps({"tool_name": tool, "tool_input": tool_input})


def _run_wrapper(
    hook: str, payload: str, *, cwd: Path, project: Path
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "CLAUDE_PROJECT_DIR": str(project),
        "HOME": str(cwd),
    }
    return subprocess.run(
        [str(WRAPPER), hook],
        input=payload,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        check=False,
    )


# --- finding 7: a qcal/ package in the working directory cannot replace the guards ----------


def test_7_shadow_package_in_the_working_directory_is_not_imported(tmp_path: Path) -> None:
    shadow = tmp_path / "qcal" / "hooks"
    shadow.mkdir(parents=True)
    (tmp_path / "qcal" / "__init__.py").write_text("")
    (shadow / "__init__.py").write_text("")
    (shadow / "__main__.py").write_text("import sys\nsys.exit(0)  # allow everything\n")

    result = _run_wrapper(
        "guard-paths",
        _payload("Write", file_path=str(REPO_ROOT / "Makefile")),
        cwd=tmp_path,
        project=REPO_ROOT,
    )

    assert result.returncode == 2, result.stderr
    assert "enforcement surface" in result.stderr


# --- finding 8: a hand-made .git marker does not move the policy root ------------------------


@pytest.mark.parametrize(
    ("marker_dir", "target"),
    [
        ("paper", "paper/sections/abstract.tex"),
        (".github", ".github/workflows/ci.yml"),
        ("runs", "runs/registry/R1.json"),
    ],
)
def test_8_fake_git_marker_cannot_unprotect_a_path(
    git_repo: Path, marker_dir: str, target: str
) -> None:
    write(git_repo, f"{marker_dir}/.git", f"gitdir: {git_repo}/.git\n")

    assert root_for(str(git_repo / target), git_repo) == git_repo
    result = _run_wrapper(
        "guard-paths",
        _payload("Edit", file_path=str(git_repo / target)),
        cwd=git_repo,
        project=git_repo,
    )
    assert result.returncode == 2, result.stderr


def test_8_marker_pointing_at_a_real_worktree_entry_still_needs_the_back_pointer(
    git_repo: Path, tmp_path: Path
) -> None:
    real = tmp_path / "wt"
    run_git(git_repo, "worktree", "add", "-q", "-b", "claude/wt", str(real))
    entry = git_repo / ".git" / "worktrees" / "wt"
    write(git_repo, "paper/.git", f"gitdir: {entry}\n")

    assert root_for(str(git_repo / "paper/sections/abstract.tex"), git_repo) == git_repo
    assert root_for(str(real / "x.py"), git_repo) == real


# --- finding 9: known spellings of a push to a protected branch are caught -------------------
# The guard is feedback; the branch ruleset (no direct pushes to main) is the control.

ALIASES = {
    ("/work", "p"): "push",
    ("/work", "pf"): "push --force",
    ("/work", "shp"): "!git push origin HEAD:main",
    ("/work", "lg"): "log --oneline",
}


def _finding(
    command: str, *, cwd: str | None = "/work", branch: str | None = "claude/x"
) -> str | None:
    found = analyze(
        command,
        protected_branches=PROTECTED,
        deny_flags=DENY_FLAGS,
        cwd=cwd,
        resolve_branch=lambda where: {"/work": branch, "/main": "main"}.get(where or ""),
        resolve_alias=lambda where, name: ALIASES.get((where or "", name)),
    )
    return found.rule if found else None


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        # adversarial review B5
        ("git -c remote.origin.push=HEAD:refs/heads/main push origin", "bash.push_unresolved"),
        ("git -c alias.p=push p origin HEAD:main", "bash.push_unresolved"),
        ("git -c Push.Default=current push", "bash.push_unresolved"),
        ("git --config-env=remote.origin.push=X push origin", "bash.push_unresolved"),
        ("GIT_DIR=/main/.git git push origin HEAD", "bash.push_unresolved"),
        ("GIT_WORK_TREE=/main git push origin @", "bash.push_unresolved"),
        ("git --git-dir=/main/.git push origin HEAD", "bash.push_unresolved"),
        ("cd -P /main && git push", "bash.push_implicit"),
        ("cd -- /main && git push", "bash.push_implicit"),
        ("(cd /work); git push", "bash.push_unresolved"),
        # found while fixing B5: shlex groups ");" so the push was never seen
        ("(cd /x); git push origin main", "bash.push_protected"),
        ("true&&(git push origin main)", "bash.push_protected"),
        # persistent aliases are resolved through git config
        ("git p origin HEAD:main", "bash.push_protected"),
        ("git pf origin claude/x", "bash.push_flag"),
        ("git shp", "bash.push_unresolved"),
    ],
)
def test_b5_configuration_aliases_and_directories_cannot_hide_a_push(
    command: str, rule: str
) -> None:
    assert _finding(command) == rule


@pytest.mark.parametrize(
    "command",
    [
        "git -c user.name=x push origin claude/x",
        "git p origin claude/x",
        "git lg",
        "git status && git log",
        "FOO=1 git push origin claude/x",
    ],
)
def test_b5_ordinary_commands_and_aliases_pass(command: str) -> None:
    assert _finding(command) is None


def test_b5_aliases_are_looked_up_with_git_config(git_repo: Path) -> None:
    from qcal.hooks.bash import git_alias

    run_git(git_repo, "config", "alias.pp", "push origin HEAD:main")
    assert git_alias(str(git_repo), "pp") == "push origin HEAD:main"
    assert git_alias(str(git_repo), "missing") is None
    found = analyze(
        "git pp",
        protected_branches=PROTECTED,
        deny_flags=DENY_FLAGS,
        cwd=str(git_repo),
        resolve_branch=lambda _: "claude/x",
    )
    assert found is not None
    assert found.rule == "bash.push_protected"


@pytest.mark.parametrize(
    "command",
    [
        "git push origin HEAD:heads/main",
        "git push origin HEAD:refs/heads/main",
        "git push origin claude/x:Heads/Main",
        "git push origin heads/civ",
    ],
)
def test_9_branch_prefix_spellings_are_protected(command: str) -> None:
    assert _finding(command) == "bash.push_protected"


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        ("git -C /main push", "bash.push_implicit"),
        ("cd /main && git push", "bash.push_implicit"),
        ("cd /main; git push origin", "bash.push_implicit"),
        ("cd $MAIN && git push", "bash.push_unresolved"),
        ("cd ~ && git push", "bash.push_unresolved"),
        ("git --git-dir=/main/.git push", "bash.push_unresolved"),
        ("git --work-tree /main push origin", "bash.push_unresolved"),
        ("cd $X && git push origin HEAD", "bash.push_unresolved"),
    ],
)
def test_9_implicit_push_from_another_directory_is_resolved_or_refused(
    command: str, rule: str
) -> None:
    assert _finding(command) == rule


@pytest.mark.parametrize(
    "command",
    [
        "git push origin claude/x",
        "git push origin HEAD:claude/x",
        "git push",
        "cd /work && git push",
        "git -C /work push",
        "cd $X && git push origin claude/x",
    ],
)
def test_9_legitimate_pushes_are_allowed(command: str) -> None:
    assert _finding(command) is None


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        # Found while writing these tests: substitution split the command so the push vanished.
        ("git -C $(pwd) push origin main", "bash.push_substitution"),
        ("git $(echo push) origin main", "bash.push_substitution"),
        ("git push origin $(echo main)", "bash.push_substitution"),
        ("git push origin `echo main`", "bash.push_substitution"),
        ("git -C <(true) push", "bash.push_substitution"),
        ("B=main; git push origin $B", "bash.push_unresolved"),
        ('git push origin "HEAD:${TARGET}"', "bash.push_unresolved"),
        ("git push origin 'refs/heads/*:refs/heads/*'", "bash.push_wildcard"),
        ("git push origin '*:*'", "bash.push_wildcard"),
        ("echo main | xargs git push origin", "bash.push_unresolved"),
        ("printf main | /usr/bin/xargs -I{} git push origin {}", "bash.push_unresolved"),
    ],
)
def test_9_dynamic_and_wildcard_pushes_are_refused(command: str, rule: str) -> None:
    assert _finding(command) == rule


@pytest.mark.parametrize(
    "command",
    [
        "echo $(date) && git status",  # substitution, but no push
        "git log --format=%H | head -1",
        "git push origin 'claude/x:claude/x'",
    ],
)
def test_9_conservative_rules_leave_unrelated_commands_alone(command: str) -> None:
    assert _finding(command) is None


def test_9_nesting_depth_is_configurable() -> None:
    nested = "bash -c \"bash -c 'git push origin main'\""
    common = {"protected_branches": PROTECTED, "deny_flags": DENY_FLAGS, "cwd": "/work"}

    assert analyze(nested, max_depth=2, resolve_branch=lambda _: None, **common) is not None
    assert analyze(nested, max_depth=0, resolve_branch=lambda _: None, **common) is None
