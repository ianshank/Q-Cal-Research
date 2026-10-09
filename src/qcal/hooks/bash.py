"""Static analysis of Bash commands for ``git push`` to protected branches.

This guard is deliberately push-only (plan §3.4). It does not try to stop file
writes from the shell: that is impossible to do reliably with pattern matching,
and the binding control for protected files is signed commits plus CI.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from qcal.log import get_logger

_log = get_logger("hooks.bash")

_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "|&", ";;", "(", ")"})
_GIT_OPTS_WITH_VALUE = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix"}
)
_PUSH_OPTS_WITH_VALUE = frozenset({"-o", "--push-option", "--repo", "--receive-pack", "--exec"})
_REFS_HEADS = "refs/heads/"
_UNPARSEABLE_PUSH = re.compile(r"\bgit\b.*\bpush\b", re.DOTALL)

BranchResolver = Callable[[str | None], str | None]


@dataclass(frozen=True)
class PushFinding:
    reason: str
    rule: str


def current_branch(cwd: str | None) -> str | None:
    """Best-effort name of the checked-out branch in ``cwd``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("cannot resolve current branch: %s", exc)
        return None
    branch = result.stdout.strip()
    return branch if result.returncode == 0 and branch and branch != "HEAD" else None


def tokenize(command: str) -> list[list[str]]:
    """Split a command line into simple commands; raises ``ValueError`` if unbalanced."""
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token in _SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [s for s in segments if s]


def analyze(
    command: str,
    *,
    protected_branches: Sequence[str],
    deny_flags: Sequence[str],
    cwd: str | None,
    resolve_branch: BranchResolver = current_branch,
    _depth: int = 0,
) -> PushFinding | None:
    """Return a finding if ``command`` pushes to a protected branch or force-pushes."""
    try:
        segments = tokenize(command)
    except ValueError:
        if _UNPARSEABLE_PUSH.search(command):
            return PushFinding(
                "cannot parse a command that mentions git push; refusing", "bash.unparseable"
            )
        return None
    protected = {b.lower() for b in protected_branches}
    for segment in segments:
        for args in _git_push_invocations(segment):
            finding = _check_push(args, protected, deny_flags, cwd, resolve_branch)
            if finding:
                return finding
        if _depth < 3:
            for token in segment:
                if "git" in token and "push" in token and (" " in token or "\t" in token):
                    nested = analyze(
                        token,
                        protected_branches=protected_branches,
                        deny_flags=deny_flags,
                        cwd=cwd,
                        resolve_branch=resolve_branch,
                        _depth=_depth + 1,
                    )
                    if nested:
                        return nested
    return None


def _git_push_invocations(segment: list[str]) -> Iterator[list[str]]:
    indexes = [i for i, token in enumerate(segment) if PurePosixPath(token).name == "git"]
    for n, start in enumerate(indexes):
        end = indexes[n + 1] if n + 1 < len(indexes) else len(segment)
        i = start + 1
        while i < end and segment[i].startswith("-"):
            i += 2 if segment[i] in _GIT_OPTS_WITH_VALUE else 1
        if i < end and segment[i] == "push":
            yield segment[i + 1 : end]


def _check_push(
    args: list[str],
    protected: set[str],
    deny_flags: Sequence[str],
    cwd: str | None,
    resolve_branch: BranchResolver,
) -> PushFinding | None:
    positionals, finding = _scan_push_args(args, deny_flags)
    if finding:
        return finding
    # The first positional is the remote, unless --repo named it; then treat every
    # positional as a refspec (conservative: over-blocking an odd command is fine).
    names_repo = any(a == "--repo" or a.startswith("--repo=") for a in args)
    refspecs = positionals if names_repo else positionals[1:]
    if not refspecs:
        branch = resolve_branch(cwd)
        if branch and branch.lower() in protected:
            return PushFinding(
                f"git push from protected branch {branch!r}; push a claude/* branch and open a PR",
                "bash.push_implicit",
            )
        return None
    for refspec in refspecs:
        finding = _check_refspec(refspec, protected, cwd, resolve_branch)
        if finding:
            return finding
    return None


def _scan_push_args(
    args: list[str], deny_flags: Sequence[str]
) -> tuple[list[str], PushFinding | None]:
    long_flags = {f for f in deny_flags if f.startswith("--")}
    short_letters = {f[1] for f in deny_flags if len(f) == 2 and f.startswith("-") and f != "--"}
    positionals: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            positionals.extend(args[i + 1 :])
            break
        if arg.startswith("--"):
            name = arg.split("=", 1)[0]
            if name in long_flags:
                return positionals, PushFinding(
                    f"git push with {name} is not allowed for agents", "bash.push_flag"
                )
            i += 2 if name in _PUSH_OPTS_WITH_VALUE and "=" not in arg else 1
        elif arg.startswith("-") and len(arg) > 1:
            hit = (
                None
                if arg in _PUSH_OPTS_WITH_VALUE
                else next((c for c in arg[1:] if c in short_letters), None)
            )
            if hit:
                return positionals, PushFinding(
                    f"git push with -{hit} is not allowed for agents", "bash.push_flag"
                )
            i += 2 if arg in _PUSH_OPTS_WITH_VALUE else 1
        else:
            positionals.append(arg)
            i += 1
    return positionals, None


def _check_refspec(
    refspec: str, protected: set[str], cwd: str | None, resolve_branch: BranchResolver
) -> PushFinding | None:
    if refspec.startswith("+"):
        return PushFinding(f"forced refspec {refspec!r} is not allowed", "bash.push_force_refspec")
    source, colon, destination = refspec.partition(":")
    target = destination if colon else source
    if target.upper() == "HEAD":
        target = resolve_branch(cwd) or target
    name = target.removeprefix(_REFS_HEADS)
    if name.lower() not in protected:
        return None
    action = "delete" if colon and not source else "push to"
    return PushFinding(
        f"agents may not {action} protected branch {name!r}; open a PR instead",
        "bash.push_protected",
    )
