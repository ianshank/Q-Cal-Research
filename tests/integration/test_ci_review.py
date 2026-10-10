"""``qcal ci review-check``: the other model's current, approving review must be committed."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from qcal.ci.review import (
    branch_slug,
    check_review,
    github_annotations,
    is_ancestor,
    required_reviewers,
    review_path,
)
from qcal.cli import main
from qcal.config import Config, load_config
from qcal.gitutil import GitError
from tests.conftest import run_git, write

pytestmark = pytest.mark.integration
BRANCH = "claude/fix-x"
REVIEW = "review/gemini/claude-fix-x.md"


def review_text(sha: str, **overrides: object) -> str:
    meta: dict[str, object] = {
        "reviewer": "gemini-3.1-pro",
        "reviewed_sha": sha,
        "verdict": "approve",
        "blocking": [],
        "non_blocking": [],
    }
    meta.update(overrides)
    return f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\nAnswers.\n"


def commit(repo: Path, message: str) -> str:
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", message)
    return run_git(repo, "rev-parse", "HEAD")


@pytest.fixture
def pr(git_repo: Path) -> tuple[Path, str, str]:
    """A base commit and a PR commit that changes code; returns (repo, base, code_sha)."""
    write(git_repo, "qcal.toml", '[review]\nmode = "enforce"\n')
    base = commit(git_repo, "policy")
    run_git(git_repo, "checkout", "-q", "-b", BRANCH)
    write(git_repo, "src/qcal_lab/x.py", "x = 1\n")
    return git_repo, base, commit(git_repo, "code")


def test_current_approving_review_passes(pr) -> None:
    repo, base, code = pr
    write(repo, REVIEW, review_text(code))
    head = commit(repo, "review")

    report = check_review(repo, base, head, BRANCH)

    assert report.passed, report.problems
    assert report.path == REVIEW
    assert report.to_dict()["verdict"] == "PASS"


def test_missing_review_fails_in_enforce_and_reports_in_bootstrap(pr) -> None:
    repo, base, code = pr

    enforce = check_review(repo, base, code, BRANCH)
    bootstrap = check_review(repo, base, code, BRANCH, mode="bootstrap")

    assert not enforce.passed
    assert enforce.problems == [f"no review file at {REVIEW}"]
    assert bootstrap.passed
    assert bootstrap.problems == enforce.problems


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"verdict": "block"}, "verdict 'block' is not 'approve'"),
        ({"reviewer": "claude-adversarial-reviewer"}, "is not 'gemini'"),
        ({"reviewed_sha": "abc123"}, "must be a full commit SHA"),
        ({"reviewed_sha": "f" * 40}, "is not a commit in this repository"),
        (
            {"blocking": [{"id": "B1", "finding": "x", "resolved_in": ""}]},
            "blocking finding B1 has no resolved_in",
        ),
        ({"blocking": "B1"}, "blocking must be a list"),
    ],
)
def test_review_content_problems(pr, overrides: dict[str, object], fragment: str) -> None:
    repo, base, code = pr
    write(repo, REVIEW, review_text(code, **overrides))
    head = commit(repo, "review")

    problems = check_review(repo, base, head, BRANCH).problems

    assert any(fragment in p for p in problems), problems


def test_resolved_blocking_findings_pass(pr) -> None:
    repo, base, code = pr
    blocking = [{"id": "B1", "finding": "x", "resolved_in": code}]
    write(repo, REVIEW, review_text(code, blocking=blocking))

    assert check_review(repo, base, commit(repo, "review"), BRANCH).passed


def test_code_changed_after_the_review_is_stale(pr) -> None:
    repo, base, code = pr
    write(repo, REVIEW, review_text(code))
    commit(repo, "review")
    write(repo, "src/qcal_lab/x.py", "x = 2  # sneaked in after review\n")
    head = commit(repo, "late change")

    problems = check_review(repo, base, head, BRANCH).problems

    assert problems == [f"{REVIEW}: changed after the review at {code[:12]}: src/qcal_lab/x.py"]


def test_review_of_a_base_commit_predates_the_pull_request(pr) -> None:
    repo, base, _code = pr
    write(repo, REVIEW, review_text(base))
    head = commit(repo, "review")

    assert "predates the pull request" in check_review(repo, base, head, BRANCH).problems[0]


def test_review_of_an_unrelated_commit_is_not_in_the_history(pr) -> None:
    repo, base, _code = pr
    run_git(repo, "checkout", "-q", "-b", "elsewhere", base)
    write(repo, "other.txt", "o\n")
    other = commit(repo, "unrelated")
    run_git(repo, "checkout", "-q", BRANCH)
    write(repo, REVIEW, review_text(other))
    head = commit(repo, "review")

    assert "not part of this pull request" in check_review(repo, base, head, BRANCH).problems[0]


@pytest.mark.parametrize("text", ["no frontmatter\n", "---\n: bad: [\n---\n"])
def test_unreadable_review_files(pr, text: str) -> None:
    repo, base, _code = pr
    write(repo, REVIEW, text)
    head = commit(repo, "review")

    assert check_review(repo, base, head, BRANCH).problems


def test_review_mode_is_validated(pr) -> None:
    repo, base, code = pr
    with pytest.raises(ValueError, match=r"review\.mode must be one of"):
        check_review(repo, base, code, BRANCH, mode="strict")


# --- configuration helpers ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("branch", "slug"),
    [
        ("claude/fix-x", "claude-fix-x"),
        ("refs/heads/gemini/a b", "gemini-a-b"),
        ("../../etc", "etc"),
        ("///", "unnamed"),
    ],
)
def test_branch_slug_is_one_safe_segment(branch: str, slug: str) -> None:
    assert branch_slug(branch) == slug
    assert "/" not in branch_slug(branch)


@pytest.mark.parametrize(
    ("branch", "reviewers"),
    [
        ("claude/x", ["gemini"]),
        ("gemini/x", ["claude"]),
        ("antigravity/x", ["claude"]),
        ("feature/x", ["claude", "gemini"]),
    ],
)
def test_required_reviewers_follow_the_branch_prefix(
    config: Config, branch: str, reviewers: list[str]
) -> None:
    assert required_reviewers(config, branch) == reviewers


def test_review_path_template_is_configurable(make_config: Callable[[str], Config]) -> None:
    config = make_config('[review]\npath_template = "reviews/{slug}.{reviewer}.md"\n')
    assert review_path(config, "gemini", "claude/a") == "reviews/claude-a.gemini.md"


def test_is_ancestor_raises_on_bad_objects(git_repo: Path) -> None:
    head = run_git(git_repo, "rev-parse", "HEAD")
    assert is_ancestor(git_repo, head, head)
    with pytest.raises(GitError):
        is_ancestor(git_repo, "nope", head)


def test_annotations(pr) -> None:
    repo, base, code = pr
    enforce = check_review(repo, base, code, BRANCH)
    bootstrap = check_review(repo, base, code, BRANCH, mode="bootstrap")

    assert github_annotations(enforce) == [f"::error::cross-review: no review file at {REVIEW}"]
    assert github_annotations(bootstrap)[-1].startswith("::notice::review.mode is 'bootstrap'")


# --- CLI ------------------------------------------------------------------------------------


def test_cli_review_check_writes_json_and_annotations(pr, tmp_path: Path) -> None:
    repo, base, code = pr
    out = io.StringIO()
    report_file = tmp_path / "review.json"

    code_ = main(
        [
            "--root",
            str(repo),
            "ci",
            "review-check",
            "--base",
            base,
            "--head",
            code,
            "--branch",
            BRANCH,
            "--github",
            "--json-out",
            str(report_file),
        ],
        out=out,
    )

    assert code_ == 1
    assert "::error::cross-review" in out.getvalue()
    assert "cross-review (enforce): FAIL" in out.getvalue()
    assert json.loads(report_file.read_text())["problems"] == [f"no review file at {REVIEW}"]


def test_cli_review_check_bad_ref_is_a_usage_error(pr) -> None:
    repo, _base, code = pr
    argv = ["--root", str(repo), "ci", "review-check", "--base", "nope", "--head", code]
    assert main([*argv, "--branch", BRANCH], out=io.StringIO()) == 2


def test_policy_comes_from_the_base_ref(pr) -> None:
    repo, base, code = pr
    write(repo, "qcal.toml", '[review]\nmode = "bootstrap"\n')  # the PR tries to relax it
    head = commit(repo, "relax review policy")

    assert not check_review(repo, base, head, BRANCH).passed
    assert load_config(repo, environ={}).str_value("review.mode") == "bootstrap"
    assert code != head
