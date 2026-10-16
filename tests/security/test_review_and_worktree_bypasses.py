"""Adversarial review B1 (cross-review self-approval) and B2 (forged worktrees)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from qcal.ci.review import check_review, names_reviewer, required_reviewers
from qcal.config import Config, load_config
from qcal.hooks import guards
from qcal.hooks.payload import HookPayload
from qcal.policy import Policy
from tests.conftest import run_git, write

pytestmark = pytest.mark.rule("A4")

# --- B1: reviews are signed, reviewers named exactly, resolutions are real commits ------------


@pytest.mark.parametrize(
    "path", ["review/claude/feat-x.md", "review/gemini/claude-x.md", "review/other/y.md"]
)
def test_b1_every_committed_review_needs_a_signature(config: Config, path: str) -> None:
    policy = Policy.from_config(config)
    assert policy.in_categories(path, config.str_list("signing.signed_categories"))


def test_b1_the_template_is_not_a_review(config: Config) -> None:
    policy = Policy.from_config(config)
    assert "cross_review" not in policy.categories_for("review/TEMPLATE.md")


def test_b1_claude_may_draft_its_own_reviews_but_never_the_other_models(
    config: Config, repo: Path
) -> None:
    policy = Policy.from_config(config)

    def decide(relative: str) -> bool:
        payload = HookPayload.parse(
            json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(repo / relative)}})
        )
        return guards.guard_paths(payload, config=config, policy=policy, root=repo).allow

    assert decide("review/claude/gemini-x.md")
    assert not decide("review/gemini/claude-x.md")


@pytest.mark.parametrize(
    ("named", "reviewer", "ok"),
    [
        ("gemini", "gemini", True),
        ("gemini-3.1-pro", "gemini", True),
        ("Gemini-2", "gemini", True),
        ("geminiX", "gemini", False),
        ("claude-adversarial-reviewer", "gemini", False),
        ("", "gemini", False),
    ],
)
def test_b1_reviewer_names_match_exactly(named: str, reviewer: str, *, ok: bool) -> None:
    assert names_reviewer(named, reviewer) is ok


@pytest.mark.parametrize(
    ("branch", "reviewer"),
    [("Claude/x", "gemini"), ("refs/heads/claude/x", "gemini"), ("GEMINI/y", "claude")],
)
def test_b1_branch_prefixes_match_case_insensitively(
    config: Config, branch: str, reviewer: str
) -> None:
    assert required_reviewers(config, branch) == [reviewer]


def _commit(repo: Path, message: str) -> str:
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", message)
    return run_git(repo, "rev-parse", "HEAD")


def _review(sha: str, resolved_in: str) -> str:
    meta = {
        "reviewer": "gemini-3.1-pro",
        "reviewed_sha": sha,
        "verdict": "approve",
        "blocking": [{"id": "B1", "finding": "f", "resolved_in": resolved_in}],
    }
    return f"---\n{yaml.safe_dump(meta)}---\n"


@pytest.fixture
def pr(git_repo: Path) -> tuple[Path, str, str, str]:
    base = run_git(git_repo, "rev-parse", "HEAD")
    run_git(git_repo, "checkout", "-q", "-b", "claude/x")
    write(git_repo, "a.py", "a = 1\n")
    fix = _commit(git_repo, "fix")
    write(git_repo, "b.py", "b = 1\n")
    reviewed = _commit(git_repo, "more")
    return git_repo, base, fix, reviewed


@pytest.mark.parametrize(
    ("which", "fragment"),
    [
        ("x", "'x' is not a commit"),
        ("base", "is not a commit of this pull request that the review covers"),
        ("later", "is not a commit of this pull request that the review covers"),
    ],
)
def test_b1_resolved_in_must_be_a_reviewed_commit_of_the_pull_request(
    pr, which: str, fragment: str
) -> None:
    repo, base, _fix, reviewed = pr
    later_path = "review/gemini/claude-x.md"
    resolved = {"x": "x", "base": base}.get(which, "")
    if which == "later":
        write(repo, "review/notes.md", "n\n")  # a review-only commit after reviewed_sha
        resolved = _commit(repo, "later")
    write(repo, later_path, _review(reviewed, resolved))
    head = _commit(repo, "review")

    problems = check_review(repo, base, head, "claude/x", mode="enforce").problems

    assert any(fragment in p for p in problems), problems


def test_b1_a_reviewed_fix_commit_resolves_the_finding(pr) -> None:
    repo, base, fix, reviewed = pr
    write(repo, "review/gemini/claude-x.md", _review(reviewed, fix))
    head = _commit(repo, "review")

    assert check_review(repo, base, head, "claude/x", mode="enforce").passed


# --- B2: git internals are never written by edit tools; symlinks are judged at their target ---


@pytest.mark.parametrize(
    "relative",
    [".git/worktrees/w/gitdir", ".git/hooks/pre-commit", "runs/.git", "paper/sub/.git"],
)
def test_b2_edit_tools_cannot_write_git_internals(
    config: Config, repo: Path, relative: str
) -> None:
    payload = HookPayload.parse(
        json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(repo / relative)}})
    )

    decision = guards.guard_paths(
        payload, config=config, policy=Policy.from_config(config), root=repo
    )

    assert decision.rule == "paths.git_internals"


def test_b2_symlink_from_a_worktree_back_into_the_project_is_judged_by_its_target(
    git_repo: Path, tmp_path: Path
) -> None:
    worktree = tmp_path / "wt"
    run_git(git_repo, "worktree", "add", "-q", "-b", "claude/wt", str(worktree))
    (worktree / "up").symlink_to(git_repo)
    config = load_config(git_repo, environ={})
    payload = HookPayload.parse(
        json.dumps(
            {"tool_name": "Edit", "tool_input": {"file_path": str(worktree / "up/src/qcal/x.py")}}
        )
    )

    decision = guards.guard_paths(
        payload, config=config, policy=Policy.from_config(config), root=git_repo
    )

    assert decision.rule == "paths.enforcement_surface"
    assert guards.trees_for(str(worktree / "up/src/qcal/x.py"), git_repo) == [worktree, git_repo]
    assert (worktree / "up").is_symlink()
