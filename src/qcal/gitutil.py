"""Thin, logged wrappers around the git CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from qcal.log import get_logger

_log = get_logger("git")

DEFAULT_TIMEOUT_S = 60.0  # mirrors git.timeout_s in defaults.toml
_timeout: ContextVar[float] = ContextVar("qcal_git_timeout", default=DEFAULT_TIMEOUT_S)


def set_timeout(seconds: float) -> None:
    """Default timeout for every git call in this context (from ``git.timeout_s``)."""
    _timeout.set(_positive(seconds))


@contextmanager
def timeout_scope(seconds: float) -> Iterator[None]:
    """Use ``seconds`` for git calls inside the block, then restore the previous value."""
    token = _timeout.set(_positive(seconds))
    try:
        yield
    finally:
        _timeout.reset(token)


def _positive(seconds: float) -> float:
    if seconds <= 0:
        raise ValueError(f"git timeout must be positive, got {seconds}")
    return seconds


class GitError(RuntimeError):
    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        super().__init__(f"git {' '.join(args)} failed ({returncode}): {stderr.strip()}")
        self.returncode = returncode
        self.stderr = stderr


def git(args: Sequence[str], cwd: Path, *, check: bool = True, timeout: float | None = None) -> str:
    _log.debug("git %s (cwd=%s)", " ".join(args), cwd)
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_timeout.get() if timeout is None else timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise GitError(args, result.returncode, result.stderr)
    return result.stdout


def try_git(args: Sequence[str], cwd: Path) -> str | None:
    """Run git and return stripped stdout, or ``None`` when git or the repo is unavailable."""
    try:
        return git(args, cwd).strip()
    except (OSError, subprocess.SubprocessError, GitError) as exc:
        _log.debug("git %s unavailable: %s", " ".join(args), exc)
        return None


def head_sha(cwd: Path) -> str | None:
    return try_git(["rev-parse", "HEAD"], cwd)


def is_dirty(cwd: Path, exclude: Sequence[str] = ()) -> bool | None:
    """Whether the work tree has changes, ignoring the repo-relative paths in ``exclude``."""
    pathspec = [":(top)", *(f":(top,exclude){p}" for p in exclude)] if exclude else []
    status = try_git(
        ["status", "--porcelain", "--", *pathspec] if pathspec else ["status", "--porcelain"], cwd
    )
    return None if status is None else bool(status)


def show_file(ref: str, path: str, cwd: Path) -> str | None:
    """Contents of ``path`` at ``ref``, or ``None`` if it does not exist there."""
    try:
        return git(["show", f"{ref}:{path}"], cwd)
    except GitError as exc:
        _log.debug("%s:%s not readable: %s", ref, path, exc.stderr.strip())
        return None


def resolve_commit(ref: str, cwd: Path) -> str:
    """Full SHA of the commit ``ref`` names; :class:`GitError` when it names none."""
    return git(
        ["rev-parse", "--verify", "--quiet", "--end-of-options", f"{ref}^{{commit}}"], cwd
    ).strip()


def merge_base(first: str, second: str, cwd: Path) -> str | None:
    """Best common ancestor of two commits, or ``None`` for unrelated histories."""
    try:
        return git(["merge-base", first, second], cwd).strip() or None
    except GitError:
        return None
