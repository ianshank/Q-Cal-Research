"""Verify that commits touching protected paths are signed by an allowed key.

Policy and ``allowed_signers`` are read from the *base* ref, never from the pull
request head, so a pull request cannot relax the rules that judge it. Merge
commits are judged only on files that differ from every parent (``--cc``), so a
routine base merge does not demand a new signature.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.config import REPO_CONFIG_NAME, Config, load_config
from qcal.gitutil import GitError, git, show_file
from qcal.log import get_logger
from qcal.policy import Policy

_log = get_logger("ci.signatures")
_MODES = ("bootstrap", "enforce")


@dataclass
class CommitVerdict:
    sha: str
    protected_files: list[str]
    signed: bool
    detail: str = ""


@dataclass
class SignatureReport:
    mode: str
    keys: int
    verdicts: list[CommitVerdict] = field(default_factory=list)

    @property
    def violations(self) -> list[CommitVerdict]:
        return [v for v in self.verdicts if v.protected_files and not v.signed]

    @property
    def passed(self) -> bool:
        return self.mode != "enforce" or not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed_keys": self.keys,
            "verdict": "PASS" if self.passed else "FAIL",
            "violations": [
                {"sha": v.sha, "files": v.protected_files, "detail": v.detail}
                for v in self.violations
            ],
            "checked_commits": len(self.verdicts),
        }


def policy_config(repo: Path, policy_ref: str) -> Config:
    text = show_file(policy_ref, REPO_CONFIG_NAME, repo)
    return (
        load_config(repo, repo_text=text, environ={})
        if text is not None
        else load_config(repo, environ={}, use_repo_file=False)
    )


def changed_files(repo: Path, sha: str) -> list[str]:
    parents = git(["rev-list", "--parents", "-n", "1", sha], repo).split()[1:]
    if len(parents) > 1:
        args = ["diff-tree", "--no-commit-id", "--name-only", "-r", "--no-renames", "--cc", sha]
    else:
        args = ["diff-tree", "--no-commit-id", "--name-only", "-r", "--no-renames", "--root", sha]
    return [line for line in git(args, repo).splitlines() if line.strip()]


def count_keys(text: str | None) -> int:
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def verify_signatures(
    repo: Path, base: str, head: str, *, policy_ref: str | None = None, mode: str | None = None
) -> SignatureReport:
    ref = policy_ref or base
    config = policy_config(repo, ref)
    chosen = mode or config.str_value("signing.mode")
    if chosen not in _MODES:
        raise ValueError(f"signing.mode must be one of {_MODES}, got {chosen!r}")
    policy = Policy.from_config(config)
    categories = config.str_list("signing.signed_categories")
    signers = show_file(ref, config.str_value("signing.allowed_signers"), repo)
    report = SignatureReport(mode=chosen, keys=count_keys(signers))
    commits = git(["rev-list", "--reverse", f"{base}..{head}"], repo).split()
    _log.info(
        "checking %d commit(s) in %s..%s with policy from %s (mode=%s, keys=%d)",
        len(commits),
        base,
        head,
        ref,
        chosen,
        report.keys,
    )

    with tempfile.TemporaryDirectory() as tmp:
        signers_file = Path(tmp) / "allowed_signers"
        signers_file.write_text(signers or "", encoding="utf-8")
        for sha in commits:
            protected = [f for f in changed_files(repo, sha) if policy.in_categories(f, categories)]
            if not protected:
                report.verdicts.append(CommitVerdict(sha, [], True, "no protected paths"))
                continue
            signed, detail = _verify_commit(repo, sha, signers_file, report.keys)
            report.verdicts.append(CommitVerdict(sha, protected, signed, detail))
    for violation in report.violations:
        _log.warning(
            "commit %s touches %s without an allowed signature (%s)",
            violation.sha[:12],
            ", ".join(violation.protected_files),
            violation.detail,
        )
    return report


def _verify_commit(repo: Path, sha: str, signers_file: Path, keys: int) -> tuple[bool, str]:
    if keys == 0:
        return False, "allowed_signers has no keys on the base ref"
    env_args = ["-c", "gpg.format=ssh", "-c", f"gpg.ssh.allowedSignersFile={signers_file}"]
    try:
        git([*env_args, "verify-commit", sha], repo)
    except GitError as exc:
        return False, exc.stderr.strip().splitlines()[-1] if exc.stderr.strip() else "unsigned"
    return True, "good signature"


def github_annotations(report: SignatureReport) -> Sequence[str]:
    """``::warning::``/``::error::`` lines for the Actions log."""
    level = "error" if report.mode == "enforce" else "warning"
    lines = [
        f"::{level}::commit {v.sha[:12]} changes protected paths ({', '.join(v.protected_files)}) "
        f"without an allowed signature: {v.detail}"
        for v in report.violations
    ]
    if report.mode == "bootstrap" and report.violations:
        lines.append(
            "::notice::signing.mode is 'bootstrap'; set it to 'enforce' in qcal.toml "
            "(in a signed commit) to make this check blocking."
        )
    return lines


__all__ = [
    "SignatureReport",
    "changed_files",
    "github_annotations",
    "policy_config",
    "verify_signatures",
]
