"""Static analysis of Bash commands for ``git push`` to protected branches.

This guard is deliberately push-only (plan §3.4). It does not try to stop file
writes from the shell: that is impossible to do reliably with pattern matching,
and the binding control for protected files is signed commits plus CI.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from qcal.config import load_defaults
from qcal.gitutil import try_git
from qcal.log import get_logger

_log = get_logger("hooks.bash")

_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "|&", ";;", "(", ")"})
# shlex groups adjacent punctuation (");", "&&(", ";("): any such run is a boundary.
_SEPARATOR_CHARS = frozenset(";&|()")
_GIT_OPTS_WITH_VALUE = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix"}
)
_PUSH_OPTS_WITH_VALUE = frozenset({"-o", "--push-option", "--repo", "--receive-pack", "--exec"})
# Ways to spell a branch as a push destination; git resolves "heads/main" to refs/heads/main.
_BRANCH_PREFIXES = ("refs/heads/", "heads/")
# A directory argument the guard cannot resolve statically (variables, home, substitution).
_UNRESOLVABLE_DIR = re.compile(r"[$`~*?\[]")
_REPO_LOCATION_OPTS = frozenset({"--git-dir", "--work-tree"})
DEFAULT_MAX_DEPTH = int(load_defaults()["hooks"]["max_nesting_depth"])
# Environment variables that point git at another repository or configuration.
_REPO_LOCATION_ENV = re.compile(r"^(?:GIT_DIR|GIT_WORK_TREE|GIT_CONFIG\w*|GIT_NAMESPACE)=")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `git -c <key>=...` keys that change what or where a push sends.
_PUSH_CONFIG = re.compile(r"^(?:remote|push|branch|url|alias)\.", re.IGNORECASE)
_CD_OPTIONS = frozenset({"-P", "-L", "-e", "-@"})
# Git ignores aliases that shadow built-in commands, so only other names are looked up.
_BUILTINS = frozenset(
    {
        "add", "am", "apply", "archive", "bisect", "blame", "branch", "cat-file", "checkout",
        "cherry-pick", "clean", "clone", "commit", "commit-tree", "config", "describe", "diff",
        "diff-tree", "fetch", "for-each-ref", "format-patch", "gc", "grep", "hash-object",
        "help", "init", "log", "ls-files", "ls-remote", "ls-tree", "merge", "merge-base", "mv",
        "notes", "pull", "read-tree", "rebase", "reflog", "remote", "reset", "restore",
        "rev-list", "rev-parse", "revert", "rm", "shortlog", "show", "show-ref",
        "sparse-checkout", "stash", "status", "submodule", "switch", "symbolic-ref", "tag",
        "update-ref", "verify-commit", "version", "worktree", "write-tree",
    }
)  # fmt: skip
_HEAD_ALIASES = frozenset({"HEAD", "@"})
_REF_SUFFIX = re.compile(r"[~^@{].*$")
_REDIRECT_CHARS = frozenset("<>&")
_MIN_ABBREVIATION = 3  # "--x": git accepts unambiguous prefixes of long options
_UNPARSEABLE_PUSH = re.compile(r"\bgit\b.*\bpush\b", re.DOTALL)
# Command substitution splits a command into pieces a static reader cannot reassemble.
_SUBSTITUTION = re.compile(r"\$\(|`|<\(|>\(")
# Characters in a refspec that make its destination unknowable or plural.
_DYNAMIC_REF = re.compile(r"[$`]")
# Runners that feed a command arguments the guard cannot see.
_ARGUMENT_FEEDERS = frozenset({"xargs", "parallel"})

BranchResolver = Callable[[str | None], str | None]
AliasResolver = Callable[[str | None, str], str | None]


@dataclass(frozen=True)
class PushFinding:
    reason: str
    rule: str


def current_branch(cwd: str | None) -> str | None:
    """Best-effort name of the checked-out branch in ``cwd``."""
    branch = try_git(["rev-parse", "--abbrev-ref", "HEAD"], Path(cwd or "."))
    return branch if branch and branch != "HEAD" else None


def git_alias(cwd: str | None, name: str) -> str | None:
    """The configured expansion of ``git <name>`` in ``cwd``, or ``None``."""
    # scrub=False: the alias the agent's own git would expand, including any defined
    # through GIT_CONFIG_* variables in its environment.
    return try_git(["config", "--get", f"alias.{name}"], Path(cwd or "."), scrub=False) or None


def tokenize(command: str) -> list[list[str]]:
    """Split a command line into simple commands; raises ``ValueError`` if unbalanced.

    Comment handling is disabled on purpose: bash treats ``#`` inside a word as a
    literal, and a lexer that drops the rest of the line would hide later commands.
    Redirections are removed so their targets are never mistaken for refspecs.
    """
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token in _SEPARATORS or (token and set(token) <= _SEPARATOR_CHARS):
            segments.append([])
        else:
            segments[-1].append(token)
    return [_strip_redirections(s) for s in segments if s]


def _is_redirect(token: str) -> bool:
    return bool(token) and set(token) <= _REDIRECT_CHARS and bool({"<", ">"} & set(token))


def _strip_redirections(tokens: list[str]) -> list[str]:
    kept: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if _is_redirect(token):
            i += 2  # the operator and its target
            continue
        if token.isdigit() and i + 1 < len(tokens) and _is_redirect(tokens[i + 1]):
            i += 1  # file-descriptor number before an operator
            continue
        kept.append(token)
        i += 1
    return kept


def analyze(
    command: str,
    *,
    protected_branches: Sequence[str],
    deny_flags: Sequence[str],
    cwd: str | None,
    resolve_branch: BranchResolver = current_branch,
    resolve_alias: AliasResolver = git_alias,
    max_depth: int = DEFAULT_MAX_DEPTH,
    _depth: int = 0,
) -> PushFinding | None:
    """Return a finding if ``command`` pushes to a protected branch or force-pushes.

    ``cd <dir>`` earlier in the command and ``git -C <dir>`` change the directory whose
    checked-out branch an implicit ``git push`` would push.
    """
    if _UNPARSEABLE_PUSH.search(command) and _SUBSTITUTION.search(command):
        return PushFinding(
            "command substitution in a command that runs git push; write the push out "
            "literally (git push origin claude/<slug>)",
            "bash.push_substitution",
        )
    try:
        segments = tokenize(command)
    except ValueError:
        if _UNPARSEABLE_PUSH.search(command):
            return PushFinding(
                "cannot parse a command that mentions git push; refusing", "bash.unparseable"
            )
        return None
    protected = {b.lower() for b in protected_branches}
    here: str | None = cwd
    # A cd inside ( ... ) does not outlive the subshell; the tokens no longer show where
    # it ended, so any cd in a command with parentheses makes the directory unknown.
    subshell = "(" in command or ")" in command
    for segment in segments:
        if segment[0] == "cd":
            here = _UNRESOLVED if subshell else _change_dir(here, _cd_target(segment))
            continue
        finding = _check_segment(
            segment,
            protected,
            deny_flags,
            here,
            resolve_branch=resolve_branch,
            resolve_alias=resolve_alias,
        )
        if finding is None and _depth < max_depth:
            finding = _check_nested(
                segment,
                protected_branches=protected_branches,
                deny_flags=deny_flags,
                cwd=here,
                resolve_branch=resolve_branch,
                resolve_alias=resolve_alias,
                max_depth=max_depth,
                depth=_depth + 1,
            )
        if finding:
            return finding
    return None


def _cd_target(segment: list[str]) -> str:
    words = [w for w in segment[1:] if w not in _CD_OPTIONS and w != "--"]
    return words[0] if words else "~"


def _check_segment(
    segment: list[str],
    protected: set[str],
    deny_flags: Sequence[str],
    here: str | None,
    *,
    resolve_branch: BranchResolver,
    resolve_alias: AliasResolver,
) -> PushFinding | None:
    for invocation in _git_push_invocations(segment, here, resolve_alias):
        if invocation.fed:
            return PushFinding(
                "git push run by xargs or a similar runner gets arguments the guard "
                "cannot see; run the push directly",
                "bash.push_unresolved",
            )
        if invocation.opaque:
            return PushFinding(
                f"git push through {invocation.opaque}, which the guard cannot follow; "
                "run git push with an explicit branch",
                "bash.push_unresolved",
            )
        finding = _check_push(
            invocation.args,
            protected,
            deny_flags,
            _apply_dirs(here, invocation.dirs) if invocation.dirs else here,
            resolve_branch,
            elsewhere=invocation.elsewhere,
        )
        if finding:
            return finding
    return None


def _check_nested(
    segment: list[str],
    *,
    protected_branches: Sequence[str],
    deny_flags: Sequence[str],
    cwd: str | None,
    resolve_branch: BranchResolver,
    resolve_alias: AliasResolver,
    max_depth: int,
    depth: int,
) -> PushFinding | None:
    """Look inside quoted arguments such as ``bash -c "git push ..."``."""
    for token in segment:
        if "git" in token and "push" in token and (" " in token or "\t" in token):
            nested = analyze(
                token,
                protected_branches=protected_branches,
                deny_flags=deny_flags,
                cwd=cwd,
                resolve_branch=resolve_branch,
                resolve_alias=resolve_alias,
                max_depth=max_depth,
                _depth=depth,
            )
            if nested:
                return nested
    return None


@dataclass(frozen=True)
class _PushInvocation:
    args: list[str]
    dirs: tuple[str, ...] = ()  # -C values, applied in order
    elsewhere: bool = False  # --git-dir/--work-tree/GIT_DIR name another repository
    fed: bool = False  # xargs and friends append arguments the guard never sees
    opaque: str = ""  # why the push cannot be followed (push config, a shell alias)


_UNRESOLVED = "\0unresolved"  # sentinel cwd: the directory cannot be known statically


def _change_dir(here: str | None, target: str) -> str:
    if here == _UNRESOLVED or _UNRESOLVABLE_DIR.search(target) or target == "-":
        return _UNRESOLVED
    return str(Path(here or ".") / target)


def _apply_dirs(here: str | None, dirs: Sequence[str]) -> str | None:
    for directory in dirs:
        here = _change_dir(here, directory)
    return here


def _git_push_invocations(
    segment: list[str], here: str | None, resolve_alias: AliasResolver
) -> Iterator[_PushInvocation]:
    indexes = [i for i, token in enumerate(segment) if PurePosixPath(token).name == "git"]
    for n, start in enumerate(indexes):
        end = indexes[n + 1] if n + 1 < len(indexes) else len(segment)
        prefix = segment[:start]
        i = start + 1
        dirs: list[str] = []
        config: dict[str, str] = {}
        elsewhere = any(_REPO_LOCATION_ENV.match(t) for t in prefix if _ENV_ASSIGNMENT.match(t))
        while i < end and segment[i].startswith("-"):
            option = segment[i].split("=", 1)[0]
            elsewhere = elsewhere or option in _REPO_LOCATION_OPTS
            if segment[i] == "-C" and i + 1 < end:
                dirs.append(segment[i + 1])
            if segment[i] == "-c" and i + 1 < end:
                key, _, value = segment[i + 1].partition("=")
                config[key.lower()] = value
            elif option == "--config-env":
                config["config-env"] = segment[i]
            i += 2 if segment[i] in _GIT_OPTS_WITH_VALUE else 1
        if i >= end:
            continue
        args = _expand_alias(segment[i], segment[i + 1 : end], config, here, resolve_alias)
        if args is None:
            continue
        fed = any(PurePosixPath(t).name in _ARGUMENT_FEEDERS for t in prefix)
        risky = sorted(k for k in config if _PUSH_CONFIG.match(k) or k == "config-env")
        opaque = f"git -c {risky[0]}" if risky else ""
        if args and args[0].startswith("!"):
            opaque = f"a shell alias for git {segment[i]}"
        yield _PushInvocation(args[1:], tuple(dirs), elsewhere, fed, opaque)


def _expand_alias(
    command: str,
    rest: list[str],
    config: dict[str, str],
    here: str | None,
    resolve_alias: AliasResolver,
) -> list[str] | None:
    """``["push", ...]`` for a push (directly or through an alias), else ``None``."""
    if command == "push":
        return [command, *rest]
    if command in _BUILTINS or command.startswith("-"):
        return None
    expansion = config.get(f"alias.{command.lower()}")
    if expansion is None and here != _UNRESOLVED:
        expansion = resolve_alias(here, command)
    if not expansion or "push" not in expansion:
        return None
    if expansion.lstrip().startswith("!"):
        return ["!", *rest]  # a shell alias: opaque
    try:
        words = shlex.split(expansion)
    except ValueError:
        return ["!", *rest]
    return [*words, *rest] if words and words[0] == "push" else ["!", *rest]


def _check_push(
    args: list[str],
    protected: set[str],
    deny_flags: Sequence[str],
    cwd: str | None,
    resolve_branch: BranchResolver,
    *,
    elsewhere: bool = False,
) -> PushFinding | None:
    positionals, finding = _scan_push_args(args, deny_flags)
    if finding:
        return finding
    # The first positional is the remote, unless --repo named it; then treat every
    # positional as a refspec (conservative: over-blocking an odd command is fine).
    names_repo = any(a == "--repo" or a.startswith("--repo=") for a in args)
    refspecs = positionals if names_repo else positionals[1:]
    if not refspecs:
        if elsewhere or cwd == _UNRESOLVED:
            return PushFinding(
                "git push without a refspec from a directory the guard cannot resolve; "
                "name the branch explicitly (git push origin claude/<slug>)",
                "bash.push_unresolved",
            )
        branch = resolve_branch(cwd)
        if branch and branch.lower() in protected:
            return PushFinding(
                f"git push from protected branch {branch!r}; push a claude/* branch and open a PR",
                "bash.push_implicit",
            )
        return None
    for refspec in refspecs:
        finding = _check_refspec(
            refspec, protected, _UNRESOLVED if elsewhere else cwd, resolve_branch
        )
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
            if name in long_flags or (
                len(name) >= _MIN_ABBREVIATION and any(f.startswith(name) for f in long_flags)
            ):
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


def _strip_branch_prefix(target: str) -> str:
    lowered = target.lower()
    for prefix in _BRANCH_PREFIXES:
        if lowered.startswith(prefix):
            return target[len(prefix) :]
    return target


def _check_refspec(
    refspec: str, protected: set[str], cwd: str | None, resolve_branch: BranchResolver
) -> PushFinding | None:
    if refspec.startswith("+"):
        return PushFinding(f"forced refspec {refspec!r} is not allowed", "bash.push_force_refspec")
    if _DYNAMIC_REF.search(refspec):
        return PushFinding(
            f"refspec {refspec!r} depends on a variable; name the branch literally",
            "bash.push_unresolved",
        )
    source, colon, destination = refspec.partition(":")
    target = destination if colon else source
    if "*" in target:
        return PushFinding(
            f"wildcard refspec {refspec!r} can push protected branches", "bash.push_wildcard"
        )
    if target.upper() in _HEAD_ALIASES or target.startswith(("@", "HEAD")):
        if cwd == _UNRESOLVED:
            return PushFinding(
                f"cannot tell which branch {target!r} names here; push an explicit branch",
                "bash.push_unresolved",
            )
        target = resolve_branch(cwd) or target
    name = _REF_SUFFIX.sub("", _strip_branch_prefix(target)) or target
    if name.lower() not in protected:
        return None
    action = "delete" if colon and not source else "push to"
    return PushFinding(
        f"agents may not {action} protected branch {name!r}; open a PR instead",
        "bash.push_protected",
    )
