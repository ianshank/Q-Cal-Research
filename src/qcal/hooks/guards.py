"""Pure guard functions: payload + configuration in, :class:`Decision` out."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from qcal.config import Config, ConfigError
from qcal.globs import first_match, literal_base
from qcal.hooks.bash import BranchResolver, analyze, current_branch
from qcal.hooks.decision import Decision
from qcal.hooks.payload import HookPayload
from qcal.policy import Policy


def project_root(config: Config, environ: Mapping[str, str]) -> Path:
    """The project directory Claude Code reports, else the configured root."""
    value = environ.get(config.str_value("hooks.project_dir_env"))
    return Path(value).resolve() if value else config.root


def root_for(path: str, default_root: Path) -> Path:
    """The git work tree containing ``path`` (a linked worktree has a ``.git`` file).

    Agents run in their own worktrees (``isolation: worktree``), which may live inside
    the project (``.claude/worktrees/x``) or beside it. Policy paths are relative to the
    work tree that holds the file, so protection follows the file, not the session.
    """
    candidate = Path(path) if Path(path).is_absolute() else default_root / path
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    return default_root


def guard_paths(payload: HookPayload, *, config: Config, policy: Policy, root: Path) -> Decision:
    """Deny edit-tool writes to protected categories and clean-room-forbidden paths."""
    tool = payload.tool_name
    if tool is None:
        return Decision.denied(
            "hook input has no tool_name; cannot check the write", "paths.no_tool"
        )
    if tool not in config.str_list("hooks.edit_tools"):
        return Decision.allowed("paths.not_edit_tool")
    paths = payload.string_fields(config.str_list("hooks.path_fields"))
    if not paths:
        return Decision.denied(
            f"{tool} call without a file path cannot be checked", "paths.no_path"
        )
    deny = config.str_list("hooks.deny_categories")
    for path in paths:
        verdict = policy.evaluate(path, root_for(path, root))
        if verdict.clean_room_hits:
            return Decision.denied(f"{path} {policy.message_for('clean_room')}", "paths.clean_room")
        blocked = verdict.blocked_by(deny)
        if blocked:
            return Decision.denied(
                f"{path} {policy.message_for(blocked[0])}", f"paths.{blocked[0]}"
            )
    return Decision.allowed("paths.ok")


def guard_bash(
    payload: HookPayload,
    *,
    config: Config,
    policy: Policy,
    resolve_branch: BranchResolver = current_branch,
) -> Decision:
    """Deny force-pushes, pushes to protected branches, and configured extra patterns."""
    command = payload.tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return Decision.allowed("bash.empty")
    hits = policy.clean_room_hits(command)
    if hits:
        return Decision.denied(
            f"command references {', '.join(hits)} and {policy.message_for('clean_room')}",
            "bash.clean_room",
        )
    for rule in config.table_list("hooks.extra_bash_deny"):
        pattern, reason = rule.get("pattern"), rule.get("reason", "matches a denied pattern")
        if not isinstance(pattern, str):
            raise ConfigError("hooks.extra_bash_deny entries need a string 'pattern'")
        if re.search(pattern, command):
            return Decision.denied(str(reason), "bash.extra")
    finding = analyze(
        command,
        protected_branches=config.str_list("git.protected_branches"),
        deny_flags=config.str_list("hooks.push_deny_flags"),
        cwd=payload.cwd,
        resolve_branch=resolve_branch,
    )
    if finding:
        return Decision.denied(finding.reason, finding.rule)
    return Decision.allowed("bash.ok")


def scope_write(
    payload: HookPayload, *, prefixes: Sequence[str], config: Config, root: Path
) -> Decision:
    """Allow edit-tool writes only below one of ``prefixes`` (repo-relative directories)."""
    paths = payload.string_fields(config.str_list("hooks.path_fields"))
    if not paths:
        return Decision.denied("write without a file path cannot be scoped", "scope.no_path")
    for path in paths:
        tree = root_for(path, root)
        allowed_roots = [(tree / p).resolve() for p in prefixes]
        target = Path(path) if Path(path).is_absolute() else root / path
        resolved = target.resolve()
        if not any(resolved == a or a in resolved.parents for a in allowed_roots):
            joined = ", ".join(prefixes)
            return Decision.denied(
                f"this agent may write only under {joined}; got {path}", "scope.outside"
            )
    return Decision.allowed("scope.ok")


def deny_read(
    payload: HookPayload, *, targets: Sequence[str], config: Config, policy: Policy, root: Path
) -> Decision:
    """Deny access to paths matching globs or whole policy categories named in ``targets``.

    Fails closed: a call without a path (Grep/Glob default to the whole project) or with
    a directory that could contain a denied file is refused; the agent must narrow it.
    """
    paths = payload.string_fields(config.str_list("hooks.path_fields"))
    if not paths:
        return Decision.denied(
            "specify a path; a project-wide search could read private files", "read.no_path"
        )
    categories = [t for t in targets if t in policy.category_names]
    globs = [t for t in targets if t not in policy.category_names]
    patterns = globs + [p for c in categories for p in policy.patterns(c)]
    for path in paths:
        tree = root_for(path, root)
        verdict = policy.evaluate(path, tree)
        if verdict.blocked_by(categories):
            return Decision.denied(f"{path} is private for this agent", "read.category")
        for rel in (verdict.relative, verdict.resolved_relative):
            if rel is None:
                continue
            if first_match(rel, globs, case_insensitive=True):
                return Decision.denied(f"{path} is private for this agent", "read.glob")
            target = tree / rel if rel else tree
            if target.is_dir() and _may_contain(rel, patterns):
                return Decision.denied(
                    f"{path} is a directory that may contain private files; narrow the path",
                    "read.directory",
                )
    return Decision.allowed("read.ok")


def _may_contain(directory: str, patterns: Sequence[str]) -> bool:
    """Whether any file below ``directory`` could match one of ``patterns``."""
    prefix = f"{directory}/" if directory else ""
    for pattern in patterns:
        base, _ = literal_base(pattern)
        if (
            not base
            or base == directory
            or base.startswith(prefix)
            or directory.startswith(f"{base}/")
        ):
            return True
    return False


def allow_only(payload: HookPayload, *, commands: Sequence[str]) -> Decision:
    """Allow a Bash call only if it is exactly one of ``commands`` (token-wise)."""
    command = payload.tool_input.get("command")
    if not isinstance(command, str):
        return Decision.denied("missing command", "allow_only.missing")
    try:
        tokens = shlex.split(command)
    except ValueError:
        return Decision.denied("command cannot be parsed", "allow_only.unparseable")
    allowed = [shlex.split(c) for c in commands]
    if tokens in allowed:
        return Decision.allowed("allow_only.ok")
    return Decision.denied(f"this agent may run only: {' | '.join(commands)}", "allow_only.denied")
