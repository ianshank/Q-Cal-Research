"""The cross-review merge rule, checked from git objects only.

A pull request merges with the *other* model's review at
``review/<reviewer>/<branch-slug>.md`` (``review.path_template``). The review's
frontmatter (``review/TEMPLATE.md``) must:

* name the required reviewer and the approving verdict;
* give a ``reviewed_sha`` that is a commit of this pull request (an ancestor of head
  that base does not already contain);
* be current: every change after ``reviewed_sha`` touches only review files, so code
  cannot change after it was reviewed;
* mark every blocking finding ``resolved_in`` a commit.

Like the signature check, policy comes from the base ref, and ``review.mode`` is
``bootstrap`` (report only) until Ian switches it to ``enforce`` in a signed commit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.ci.signatures import changed_between, policy_config, require_commit
from qcal.config import Config
from qcal.gitutil import GitError, git, show_file
from qcal.globs import first_match
from qcal.integrity.licenses import FrontmatterError, read_frontmatter
from qcal.log import get_logger
from qcal.reports import verdict

_log = get_logger("ci.review")
_MODES = ("bootstrap", "enforce")
_FULL_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ReviewReport:
    mode: str
    branch: str
    reviewers: list[str]
    path: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.mode != "enforce" or not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "verdict": verdict(self.passed),
            "branch": self.branch,
            "required_reviewers": self.reviewers,
            "review_file": self.path,
            "problems": self.problems,
        }

    def render_text(self) -> str:
        head = f"cross-review ({self.mode}): {'PASS' if self.passed else 'FAIL'} for {self.branch}"
        lines = [head, f"  review file: {self.path or 'none found'}"]
        lines += [f"  problem: {p}" for p in self.problems]
        return "\n".join(lines)


def branch_slug(branch: str) -> str:
    """``claude/fix-x`` -> ``claude-fix-x``: one path segment, safe in any file system."""
    return _SLUG_UNSAFE.sub("-", branch.removeprefix("refs/heads/")).strip("-.") or "unnamed"


def required_reviewers(config: Config, branch: str) -> list[str]:
    """The model that must review ``branch``: by branch prefix, else any configured reviewer."""
    by_prefix = config.section("review.reviewer_by_branch_prefix")
    for prefix, reviewer in sorted(by_prefix.items(), key=lambda kv: -len(str(kv[0]))):
        if branch.startswith(str(prefix)):
            return [str(reviewer)]
    return config.str_list("review.reviewers")


def review_path(config: Config, reviewer: str, branch: str) -> str:
    return config.str_value("review.path_template").format(
        reviewer=reviewer, slug=branch_slug(branch)
    )


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    try:
        git(["merge-base", "--is-ancestor", ancestor, descendant], repo)
    except GitError as exc:
        if exc.returncode == 1:
            return False
        raise
    return True


def check_review(
    repo: Path,
    base: str,
    head: str,
    branch: str,
    *,
    policy_ref: str | None = None,
    mode: str | None = None,
) -> ReviewReport:
    base = require_commit(repo, base, "base")
    head = require_commit(repo, head, "head")
    config = policy_config(repo, policy_ref or base)
    chosen = mode or config.str_value("review.mode")
    if chosen not in _MODES:
        raise ValueError(f"review.mode must be one of {_MODES}, got {chosen!r}")
    reviewers = required_reviewers(config, branch)
    report = ReviewReport(mode=chosen, branch=branch, reviewers=reviewers)
    candidates = [(r, review_path(config, r, branch)) for r in reviewers]
    for reviewer, path in candidates:
        text = show_file(head, path, repo)
        if text is None:
            continue
        report.path = path
        report.problems = _judge(
            repo, config, base=base, head=head, reviewer=reviewer, path=path, text=text
        )
        if not report.problems:
            break
    if report.path is None:
        wanted = " or ".join(path for _, path in candidates)
        report.problems.append(f"no review file at {wanted}")
    for problem in report.problems:
        _log.warning("cross-review: %s", problem)
    return report


def _judge(
    repo: Path, config: Config, *, base: str, head: str, reviewer: str, path: str, text: str
) -> list[str]:
    try:
        meta = read_frontmatter(text)
    except FrontmatterError as exc:
        return [f"{path}: {exc}"]
    if not meta:
        return [f"{path}: no YAML frontmatter (see review/TEMPLATE.md)"]
    problems: list[str] = []
    named = str(meta.get("reviewer", ""))
    if not named.lower().startswith(reviewer.lower()):
        problems.append(f"{path}: reviewer {named!r} is not {reviewer!r}")
    wanted = config.str_value("review.approve_verdict")
    if str(meta.get("verdict", "")).strip().lower() != wanted:
        problems.append(f"{path}: verdict {meta.get('verdict')!r} is not {wanted!r}")
    problems += _open_blocking(path, meta.get("blocking"))
    problems += _currency(
        repo, config, base=base, head=head, path=path, reviewed=str(meta.get("reviewed_sha", ""))
    )
    return problems


def _open_blocking(path: str, blocking: Any) -> list[str]:
    if blocking in (None, []):
        return []
    if not isinstance(blocking, list):
        return [f"{path}: blocking must be a list"]
    open_ids = [
        str(item.get("id", i + 1)) if isinstance(item, dict) else str(i + 1)
        for i, item in enumerate(blocking)
        if not (isinstance(item, dict) and str(item.get("resolved_in") or "").strip())
    ]
    return [f"{path}: blocking findings without resolved_in: {open_ids}"] if open_ids else []


def _currency(
    repo: Path, config: Config, *, base: str, head: str, path: str, reviewed: str
) -> list[str]:
    if not _FULL_SHA.fullmatch(reviewed):
        return [f"{path}: reviewed_sha must be a full commit SHA, got {reviewed!r}"]
    try:
        reviewed = require_commit(repo, reviewed, "reviewed_sha")
    except ValueError:
        return [f"{path}: reviewed_sha {reviewed[:12]} is not a commit in this repository"]
    if not is_ancestor(repo, reviewed, head):
        return [f"{path}: reviewed_sha {reviewed[:12]} is not part of this pull request's history"]
    if is_ancestor(repo, reviewed, base):
        return [f"{path}: reviewed_sha {reviewed[:12]} predates the pull request"]
    allowed = config.str_list("review.post_review_globs")
    late = [p for p in changed_between(repo, reviewed, head) if first_match(p, allowed) is None]
    if late:
        shown = ", ".join(late[:5]) + (" ..." if len(late) > 5 else "")
        return [f"{path}: changed after the review at {reviewed[:12]}: {shown}"]
    return []


def github_annotations(report: ReviewReport) -> list[str]:
    level = "error" if report.mode == "enforce" else "warning"
    lines = [f"::{level}::cross-review: {p}" for p in report.problems]
    if report.mode == "bootstrap" and report.problems:
        lines.append(
            "::notice::review.mode is 'bootstrap'; set it to 'enforce' in qcal.toml "
            "(in a signed commit) to make this check blocking."
        )
    return lines


__all__ = [
    "ReviewReport",
    "branch_slug",
    "check_review",
    "github_annotations",
    "is_ancestor",
    "required_reviewers",
    "review_path",
]
