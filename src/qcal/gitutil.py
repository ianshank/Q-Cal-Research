"""Thin, logged wrappers around the git CLI.

Git runs with a scrubbed environment (:func:`git_environment`). The registry and the CI
checks ask git what ``HEAD`` holds and whether the tree is dirty; variables such as
``GIT_DIR``, ``GIT_INDEX_FILE`` or ``GIT_CONFIG_COUNT`` would let the caller's environment
answer instead of the repository, and ``git replace`` objects would let a local ref swap a
committed blob. Every call also disables ``core.fsmonitor``, a configured command that git
would run and whose answer it would trust about which files changed.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Final

from qcal.config import load_defaults
from qcal.log import get_logger

_log = get_logger("git")

#: ``GIT_*`` variables that pass through: identity and the switches that make git ignore
#: system or user configuration. Every other ``GIT_*`` variable is dropped.
GIT_ENV_KEEP: Final = frozenset(
    {
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_DATE",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_COMMITTER_DATE",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_TERMINAL_PROMPT",
    }
)
#: Set on every call: replacement objects are ignored, so ``HEAD:<path>`` is the commit's blob.
GIT_ENV_SET: Final = {"GIT_NO_REPLACE_OBJECTS": "1"}
#: Prepended to every command line; command-line configuration overrides every config file.
GIT_SAFE_CONFIG: Final = ("-c", "core.fsmonitor=false")


def git_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment git runs with: ``environ`` minus redirecting ``GIT_*`` variables."""
    source = os.environ if environ is None else environ
    env = {k: v for k, v in source.items() if not k.startswith("GIT_") or k in GIT_ENV_KEEP}
    dropped = sorted(k for k in source if k.startswith("GIT_") and k not in GIT_ENV_KEEP)
    if dropped:
        _log.debug("git runs without %s", ", ".join(dropped))
    env.update(GIT_ENV_SET)
    return env


DEFAULT_TIMEOUT_S = float(load_defaults()["git"]["timeout_s"])
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
        ["git", *GIT_SAFE_CONFIG, *args],
        cwd=cwd,
        env=git_environment(),
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
