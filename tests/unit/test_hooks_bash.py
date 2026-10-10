"""Push analysis, including every bypass and false-positive case from plan §10.3."""

from __future__ import annotations

from pathlib import Path

import pytest

from qcal.hooks.bash import analyze, current_branch, tokenize

PROTECTED = ["main", "civ"]
DENY_FLAGS = [
    "--force",
    "-f",
    "--force-with-lease",
    "--force-if-includes",
    "--mirror",
    "--all",
    "--delete",
    "-d",
    "--prune",
]


def run(command: str, branch: str | None = "claude/feature") -> str | None:
    finding = analyze(
        command,
        protected_branches=PROTECTED,
        deny_flags=DENY_FLAGS,
        cwd=None,
        resolve_branch=lambda _cwd: branch,
    )
    return finding.rule if finding else None


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        ("git push --force origin claude/x", "bash.push_flag"),
        ("git push -f origin claude/x", "bash.push_flag"),
        ("git push -uf origin claude/x", "bash.push_flag"),
        ("git push --force-with-lease=claude/x origin claude/x", "bash.push_flag"),
        ("git push --force-if-includes origin claude/x", "bash.push_flag"),
        ("git push --mirror origin", "bash.push_flag"),
        ("git push --all origin", "bash.push_flag"),
        ("git push --delete origin claude/x", "bash.push_flag"),
        ("git push origin HEAD:main", "bash.push_protected"),
        ("git push -u origin main", "bash.push_protected"),
        ("git push origin +main", "bash.push_force_refspec"),
        ("git push origin +claude/x", "bash.push_force_refspec"),
        ("git push origin refs/heads/main", "bash.push_protected"),
        ("git push origin claude/x:refs/heads/main", "bash.push_protected"),
        ("git push origin :main", "bash.push_protected"),
        ("git push origin MAIN", "bash.push_protected"),
        ("git push origin civ", "bash.push_protected"),
        ("git -C /tmp/x push origin main", "bash.push_protected"),
        ("git -c user.name=x push origin main", "bash.push_protected"),
        ("/usr/bin/git push origin main", "bash.push_protected"),
        ("sudo git push origin main", "bash.push_protected"),
        ("env A=1 git push origin main", "bash.push_protected"),
        ("cd repo && git push origin main", "bash.push_protected"),
        ("git status; git push origin main", "bash.push_protected"),
        ("git status\ngit push origin main", "bash.push_protected"),
        ("true || git push origin main", "bash.push_protected"),
        ("bash -c 'git push --force origin x'", "bash.push_flag"),
        ('sh -c "cd r && git push origin main"', "bash.push_protected"),
        ("git push -o ci.skip origin main", "bash.push_protected"),
        ("git push --repo origin main", "bash.push_protected"),
        ('git push "origin main', "bash.unparseable"),
        # code-review regressions: comment char mid-word, redirections, abbreviations, @ alias
        ("echo a#b; git push --force origin main", "bash.push_flag"),
        ("echo a#b; git push origin main", "bash.push_protected"),
        ("git push origin main>/dev/null", "bash.push_protected"),
        ("git push origin main 2>&1", "bash.push_protected"),
        ("git push --mirr origin", "bash.push_flag"),
        ("git push --al origin", "bash.push_flag"),
        ("git push --forc origin claude/x", "bash.push_flag"),
        ("git push origin main~0", "bash.push_protected"),
        ("git push origin main^0:main", "bash.push_protected"),
    ],
)
def test_denied_commands(command: str, rule: str) -> None:
    assert run(command) == rule


@pytest.mark.parametrize(
    "command",
    [
        "git push origin claude/feature",
        "git push -u origin claude/feature",
        "git push origin main-fix",
        "git push origin claude/main",
        "git push origin HEAD",
        "git push",
        "git push --tags origin",
        "git push -o ci.skip origin claude/x",
        "git push --push-option=ci.skip origin claude/x",
        "pytest --device cpu > log.txt && cat EXPERIMENTS.yaml",
        "cp a b && cat handwritten/x.py",
        "python -m pytest tests --format=json",
        "git commit -m 'push to main later'",
        "git log --oneline main",
        'echo "unbalanced',
        "git push origin claude/x 2>&1 | tee push.log",
        "git push --dry-run origin claude/x",
        "",
    ],
)
def test_allowed_commands(command: str) -> None:
    assert run(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "git push",
        "git push origin",
        "git push origin HEAD",
        "git push origin @",
        "git push origin 2>&1",
        "git push origin >/dev/null",
        "git push origin HEAD~0:HEAD",
    ],
)
def test_implicit_pushes_from_a_protected_branch_are_denied(command: str) -> None:
    assert run(command, branch="main") in {"bash.push_implicit", "bash.push_protected"}


def test_implicit_push_with_unknown_branch_is_allowed() -> None:
    assert run("git push", branch=None) is None


def test_echoed_push_text_is_treated_conservatively() -> None:
    # A quoted command string that would push to main is flagged even inside echo.
    assert run('echo "git push origin main"') == "bash.push_protected"


def test_tokenize_splits_on_shell_operators() -> None:
    assert tokenize("a b; c && d | e\nf") == [["a", "b"], ["c"], ["d"], ["e"], ["f"]]
    with pytest.raises(ValueError, match="quotation"):
        tokenize('echo "x')


def test_current_branch_in_a_repository(git_repo: Path) -> None:
    assert current_branch(str(git_repo)) == "main"


def test_current_branch_outside_a_repository(tmp_path: Path) -> None:
    assert current_branch(str(tmp_path)) is None


def test_current_branch_detached_head_is_none(git_repo: Path) -> None:
    from tests.conftest import run_git

    run_git(git_repo, "checkout", "-q", "--detach")
    assert current_branch(str(git_repo)) is None


def test_current_branch_survives_missing_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("no git")

    monkeypatch.setattr("qcal.gitutil.subprocess.run", boom)
    assert current_branch(str(tmp_path)) is None
