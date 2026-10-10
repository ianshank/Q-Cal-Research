"""Thin, logged wrappers around the git CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from qcal.log import get_logger

_log = get_logger("git")


class GitError(RuntimeError):
    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        super().__init__(f"git {' '.join(args)} failed ({returncode}): {stderr.strip()}")
        self.returncode = returncode
        self.stderr = stderr


def git(args: Sequence[str], cwd: Path, *, check: bool = True, timeout: float = 60) -> str:
    _log.debug("git %s (cwd=%s)", " ".join(args), cwd)
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
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
    except GitError:
        return None
