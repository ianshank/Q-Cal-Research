"""Verify that changes to protected paths come from commits signed by an allowed key.

Policy and ``allowed_signers`` are read from the *base* ref, never from the pull
request head, so a pull request cannot relax the rules that judge it.

Two rules, both required:

* **Per commit.** Every non-merge commit in ``base..head`` that changes a protected
  path relative to its parent must carry a good SSH signature from an allowed key.
* **Net content.** Every protected path whose content differs between ``base`` and
  ``head`` must hold content that a *signed* commit in ``base..head`` introduced
  (changed relative to its first parent). Merges, and commits built on old history,
  can therefore only carry forward content that Ian signed or that base already had;
  they cannot delete or roll back protected files.

All git path output is NUL-delimited (``-z``), so unusual file names cannot dodge the
patterns by being quoted.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.config import REPO_CONFIG_NAME, Config, load_config
from qcal.gitutil import GitError, git, merge_base, resolve_commit, show_file
from qcal.log import get_logger
from qcal.policy import Policy
from qcal.reports import verdict

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
    net: list[CommitVerdict] = field(default_factory=list)

    @property
    def violations(self) -> list[CommitVerdict]:
        per_commit = [v for v in self.verdicts if v.protected_files and not v.signed]
        covered = {f for v in per_commit for f in v.protected_files}
        net = [v for v in self.net if not v.signed and not set(v.protected_files) <= covered]
        return per_commit + net

    @property
    def passed(self) -> bool:
        return self.mode != "enforce" or not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed_keys": self.keys,
            "verdict": verdict(self.passed),
            "violations": [
                {"sha": v.sha, "files": v.protected_files, "detail": v.detail}
                for v in self.violations
            ],
            "checked_commits": len(self.verdicts),
            "checked_net_paths": len(self.net),
        }

    def render_text(self) -> str:
        lines = [
            (
                f"signatures ({self.mode}, {self.keys} allowed key(s)): {verdict(self.passed)}; "
                f"{len(self.verdicts)} commit(s), {len(self.net)} protected path(s) changed"
            )
        ]
        lines += [
            f"  {v.sha[:12]} {', '.join(v.protected_files)}: {v.detail}" for v in self.violations
        ]
        return "\n".join(lines)


class RefError(ValueError):
    """A ref given to a CI check names no commit (a usage error, never a silent fallback)."""


def require_commit(repo: Path, ref: str, role: str) -> str:
    try:
        return resolve_commit(ref, repo)
    except GitError:
        raise RefError(f"{role} ref {ref!r} does not name a commit in {repo}") from None


def policy_config(repo: Path, policy_ref: str) -> Config:
    """Policy as committed at ``policy_ref``; packaged defaults when that commit has no qcal.toml.

    A ref that names no commit is an error: falling back to defaults there would turn
    an enforcing policy into ``bootstrap`` because of a typo.
    """
    require_commit(repo, policy_ref, "policy")
    text = show_file(policy_ref, REPO_CONFIG_NAME, repo)
    if text is None:
        _log.warning(
            "%s has no %s; judging with the packaged defaults", policy_ref, REPO_CONFIG_NAME
        )
        return load_config(repo, environ={}, use_repo_file=False)
    return load_config(repo, repo_text=text, environ={})


def _nul_split(output: str) -> list[str]:
    return [part for part in output.split("\0") if part]


def parents_of(repo: Path, sha: str) -> list[str]:
    return git(["rev-list", "--parents", "-n", "1", sha], repo).split()[1:]


def changed_files(repo: Path, sha: str) -> list[str]:
    """Paths a non-merge commit changes relative to its parent (or all paths for a root)."""
    args = ["diff-tree", "-z", "--no-commit-id", "--name-only", "-r", "--no-renames", "--root", sha]
    return _nul_split(git(args, repo))


def changed_between(repo: Path, base: str, head: str) -> list[str]:
    return _changed_vs(repo, base, head)


def _changed_vs(repo: Path, old: str, new: str) -> list[str]:
    return _nul_split(git(["diff", "-z", "--name-only", "--no-renames", old, new], repo))


def blob_id(repo: Path, ref: str, path: str) -> str | None:
    """Blob id of ``path`` at ``ref``, or ``None`` when the path does not exist there."""
    try:
        return git(["rev-parse", "--verify", "--quiet", f"{ref}:{path}"], repo).strip() or None
    except GitError:
        return None


def tree_entries(repo: Path, ref: str, *, with_mode: bool = False) -> dict[str, str]:
    """Every path in ``ref``'s tree mapped to its object id, or ``"<mode> <id>"`` with
    ``with_mode`` (one ``ls-tree`` call, ``-z``)."""
    entries: dict[str, str] = {}
    for record in _nul_split(git(["ls-tree", "-r", "-z", "--full-tree", ref], repo)):
        meta, _, path = record.partition("\t")
        mode, _kind, oid = meta.split()
        entries[path] = f"{mode} {oid}" if with_mode else oid
    return entries


class TreeIndex:
    """Memoised ``blob_id`` over many refs: one ``ls-tree`` per ref instead of one call per path.

    The net-content rule asks for the same few paths at every commit in the range and
    its first parent, which was ``O(commits x paths)`` subprocesses before this cache.
    """

    def __init__(self, repo: Path) -> None:
        self._repo = repo
        self._trees: dict[str, dict[str, str]] = {}

    def entry(self, ref: str, path: str) -> str | None:
        """``"<mode> <object id>"`` of ``path`` at ``ref``: content and executable bit."""
        if ref not in self._trees:
            self._trees[ref] = tree_entries(self._repo, ref, with_mode=True)
        return self._trees[ref].get(path)

    def blob(self, ref: str, path: str) -> str | None:
        entry = self.entry(ref, path)
        return entry.split(" ", 1)[1] if entry else None

    @property
    def refs_loaded(self) -> int:
        return len(self._trees)


def count_keys(text: str | None) -> int:
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def verify_signatures(
    repo: Path, base: str, head: str, *, policy_ref: str | None = None, mode: str | None = None
) -> SignatureReport:
    base = require_commit(repo, base, "base")
    head = require_commit(repo, head, "head")
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

    require_match = config.bool_value("signing.require_committer_match")
    ssh_program = config.str_value("signing.ssh_program")
    with tempfile.TemporaryDirectory() as tmp:
        signers_file = Path(tmp) / "allowed_signers"
        signers_file.write_text(signers or "", encoding="utf-8")
        cache: dict[str, tuple[bool, str]] = {}

        def signed(sha: str) -> tuple[bool, str]:
            if sha not in cache:
                cache[sha] = _verify_commit(
                    repo,
                    sha,
                    signers_file,
                    report.keys,
                    require_match=require_match,
                    ssh_program=ssh_program,
                )
            return cache[sha]

        parents = {sha: parents_of(repo, sha) for sha in commits}
        trees = TreeIndex(repo)
        for sha in commits:
            if len(parents[sha]) > 1:
                # A merge may carry content from any parent; the net-content rule below
                # decides whether what it brings in was authored by a signed commit.
                merged = _changed_vs(repo, parents[sha][0], sha)
                touched = [
                    f
                    for f in merged
                    if policy.in_categories(f, categories)
                    and trees.entry(sha, f) != trees.entry(base, f)  # base merges are fine
                ]
                detail = "merge: judged by net content" if touched else "no protected paths"
                report.verdicts.append(CommitVerdict(sha, touched, signed=True, detail=detail))
                continue
            protected = [f for f in changed_files(repo, sha) if policy.in_categories(f, categories)]
            if not protected:
                report.verdicts.append(
                    CommitVerdict(sha, [], signed=True, detail="no protected paths")
                )
                continue
            ok, detail = signed(sha)
            report.verdicts.append(CommitVerdict(sha, protected, ok, detail))

        # Judge what this branch changed, not what base gained since it branched: a pull
        # request that is merely behind base must not fail for base's own signed edits.
        # Merges inside the range are still diffed against this fork point, so a merge
        # that drops or rolls back protected content is caught.
        fork_point = merge_base(base, head, repo) or base
        for path in changed_between(repo, fork_point, head):
            if not policy.in_categories(path, categories):
                continue
            # Mode is content: a merge that only drops +x from a hook is a change too.
            target = trees.entry(head, path)
            authors = [
                sha
                for sha in commits
                if trees.entry(sha, path) == target
                and (not parents[sha] or trees.entry(parents[sha][0], path) != target)
            ]
            good = next((sha for sha in authors if signed(sha)[0]), None)
            detail = (
                f"content introduced by signed commit {good[:12]}"
                if good
                else "net change not introduced by any signed commit in the range"
            )
            report.net.append(CommitVerdict(head, [path], good is not None, detail))
        _log.debug("net-content rule read %d tree(s)", trees.refs_loaded)
    for violation in report.violations:
        _log.warning(
            "commit %s touches %s without an allowed signature (%s)",
            violation.sha[:12],
            ", ".join(violation.protected_files),
            violation.detail,
        )
    return report


def _verify_commit(
    repo: Path, sha: str, signers_file: Path, keys: int, *, require_match: bool, ssh_program: str
) -> tuple[bool, str]:
    if keys == 0:
        return False, "allowed_signers has no keys on the base ref"
    kind = signature_kind(repo, sha)
    if kind == "no":
        return False, "unsigned"
    if kind != "ssh":
        return False, f"{kind} signature; only SSH signatures from allowed_signers count"
    env_args = [
        "-c",
        "gpg.format=ssh",
        "-c",
        f"gpg.ssh.program={ssh_program}",
        "-c",
        f"gpg.ssh.allowedSignersFile={signers_file}",
        # Never let git fall back to another verifier chosen by the signature itself.
        "-c",
        "gpg.openpgp.program=false",
        "-c",
        "gpg.x509.program=false",
    ]
    try:
        git([*env_args, "verify-commit", sha], repo)
    except GitError as exc:
        return False, exc.stderr.strip().splitlines()[-1] if exc.stderr.strip() else "unsigned"
    if require_match:
        signer, _, committer = (
            git([*env_args, "log", "-1", "--format=%GS%x00%ce", sha], repo)
            .strip()
            .partition("\x00")
        )
        if signer.lower() != committer.lower():
            return False, f"signed by {signer!r} but committed as {committer!r}"
    return True, "good signature"


_SIGNATURE_HEADERS = {
    "-----BEGIN SSH SIGNATURE-----": "ssh",
    "-----BEGIN PGP SIGNATURE-----": "openpgp",
    "-----BEGIN SIGNED MESSAGE-----": "x509",
}


def signature_kind(repo: Path, sha: str) -> str:
    """``ssh``, ``openpgp``, ``x509``, ``unknown`` or ``no`` from the raw commit object."""
    header = None
    for line in git(["cat-file", "commit", sha], repo).splitlines():
        if not line:
            break  # end of headers
        if line.startswith(("gpgsig ", "gpgsig-sha256 ")):
            header = line.split(" ", 1)[1].strip()
            break
    if header is None:
        return "no"
    return _SIGNATURE_HEADERS.get(header, "unknown")


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
    "RefError",
    "SignatureReport",
    "TreeIndex",
    "blob_id",
    "changed_between",
    "changed_files",
    "github_annotations",
    "parents_of",
    "policy_config",
    "require_commit",
    "signature_kind",
    "tree_entries",
    "verify_signatures",
]
