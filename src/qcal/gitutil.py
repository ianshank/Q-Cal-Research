"""Thin, logged wrappers around the git CLI.

Git runs with a scrubbed environment (:func:`git_environment`). The registry and the CI
checks ask git what ``HEAD`` holds and whether the tree is dirty; variables such as
``GIT_DIR``, ``GIT_INDEX_FILE`` or ``GIT_CONFIG_COUNT`` would let the caller's environment
answer instead of the repository, and ``git replace`` objects would let a local ref swap a
committed blob. Every call also disables ``core.fsmonitor``, a configured command that git
would run and whose answer it would trust about which files changed.

The registry's own questions (``HEAD``, a committed blob, the top level, a dirty tree) go
further with ``isolated=True``: no global or system configuration, no global ignore or
attributes file, the C locale. A ``GIT_CONFIG_GLOBAL`` or ``XDG_CONFIG_HOME`` that hides
untracked files cannot then make a dirty tree look clean. The alias lookup of the guard
hooks is the exception: it must see what the agent's own git would see (``scrub=False``).
"""

from __future__ import annotations

import hashlib
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
#: Added for the registry's own queries: user and system configuration play no part.
ISOLATED_ENV: Final = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "LC_ALL": "C",
}
ISOLATED_CONFIG: Final = (
    "-c",
    f"core.excludesFile={os.devnull}",
    "-c",
    f"core.attributesFile={os.devnull}",
)
GIT_MARKER: Final = ".git"


def git_environment(
    environ: Mapping[str, str] | None = None, *, isolated: bool = False
) -> dict[str, str]:
    """The environment git runs with: ``environ`` minus redirecting ``GIT_*`` variables."""
    source = os.environ if environ is None else environ
    env = {k: v for k, v in source.items() if not k.startswith("GIT_") or k in GIT_ENV_KEEP}
    dropped = sorted(k for k in source if k.startswith("GIT_") and k not in GIT_ENV_KEEP)
    if dropped:
        _log.debug("git runs without %s", ", ".join(dropped))
    env.update(GIT_ENV_SET)
    if isolated:
        env.update(ISOLATED_ENV)
    return env


def _argv(args: Sequence[str], *, isolated: bool) -> list[str]:
    return ["git", *GIT_SAFE_CONFIG, *(ISOLATED_CONFIG if isolated else ()), *args]


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


def git(
    args: Sequence[str],
    cwd: Path,
    *,
    check: bool = True,
    timeout: float | None = None,
    isolated: bool = False,
    scrub: bool = True,
) -> str:
    """Run git. ``isolated``: no user or system configuration (the registry's queries).
    ``scrub=False``: the caller's environment unchanged (what the caller's own git sees)."""
    _log.debug("git %s (cwd=%s)", " ".join(args), cwd)
    result = subprocess.run(
        _argv(args, isolated=isolated),
        cwd=cwd,
        env=git_environment(isolated=isolated) if scrub else dict(os.environ),
        capture_output=True,
        text=True,
        timeout=_timeout.get() if timeout is None else timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise GitError(args, result.returncode, result.stderr)
    return result.stdout


def try_git(
    args: Sequence[str], cwd: Path, *, isolated: bool = False, scrub: bool = True
) -> str | None:
    """Run git and return stripped stdout, or ``None`` when git or the repo is unavailable."""
    try:
        return git(args, cwd, isolated=isolated, scrub=scrub).strip()
    except (OSError, subprocess.SubprocessError, GitError) as exc:
        _log.debug("git %s unavailable: %s", " ".join(args), exc)
        return None


def head_sha(cwd: Path) -> str | None:
    return try_git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd, isolated=True) or None


def toplevel(cwd: Path) -> Path | None:
    """The work tree's top level as git sees it (isolated), or ``None``."""
    top = try_git(["rev-parse", "--show-toplevel"], cwd, isolated=True)
    return Path(top).resolve() if top else None


def git_marker(path: Path) -> Path | None:
    """The nearest ``.git`` (directory or file) at ``path`` or above it, found on disk.

    Independent of git itself: a repository that git cannot read (wrong owner, a broken
    ``.git`` file, a missing binary) is still noticed.
    """
    for candidate in (path, *path.parents):
        marker = candidate / GIT_MARKER
        if marker.exists():
            return marker
    return None


def blob_sha256(ref: str, path: str, cwd: Path) -> str | None:
    """sha256 of the bytes of ``path`` at ``ref`` (isolated, no text decoding), or ``None``."""
    try:
        result = subprocess.run(
            _argv(["cat-file", "blob", f"{ref}:{path}"], isolated=True),
            cwd=cwd,
            env=git_environment(isolated=True),
            capture_output=True,
            timeout=_timeout.get(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("git cat-file %s:%s unavailable: %s", ref, path, exc)
        return None
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def is_dirty(cwd: Path, exclude: Sequence[str] = ()) -> bool | None:
    """Whether the work tree has changes, ignoring the repo-relative paths in ``exclude``.

    Untracked files count whatever ``status.showUntrackedFiles`` says, and only the
    repository's own ignore files apply (isolated).
    """
    pathspec = [":(top)", *(f":(top,exclude){p}" for p in exclude)] if exclude else []
    status = try_git(
        ["status", "--porcelain", "--untracked-files=all", "--", *pathspec]
        if pathspec
        else ["status", "--porcelain", "--untracked-files=all"],
        cwd,
        isolated=True,
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
