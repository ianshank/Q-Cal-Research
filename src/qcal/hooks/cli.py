"""Dispatcher for ``python -m qcal.hooks <hook> [args]``.

Guards fail closed: malformed input or an internal error is a denial. The
``claims`` Stop hook fails open, because CI runs the same check as a hard gate
and a crashing Stop hook would otherwise trap the session in a loop.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO

from qcal import gitutil
from qcal.config import Config, find_root, load_config
from qcal.hooks import guards
from qcal.hooks.decision import BLOCK_EXIT_CODE, Decision, Mode, emit, resolve_mode
from qcal.hooks.payload import HookPayload, PayloadError
from qcal.log import configure_logging, get_logger
from qcal.policy import Policy

_log = get_logger("hooks.cli")

GUARD_HOOKS = ("guard-paths", "guard-bash", "scope-write", "deny-read", "allow-only")
STOP_HOOKS = ("claims",)
# Set by run_hook.sh: "1" when the wrapper runs the hook fail-open (the Stop hook). The hook
# applies the same mode to its own argument errors, so a typo cannot trap a session.
FAIL_OPEN_ENV = "QCAL_HOOK_FAIL_OPEN"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qcal.hooks", description=__doc__)
    sub = parser.add_subparsers(dest="hook", required=True)
    sub.add_parser("guard-paths", help="deny edits to protected and clean-room paths")
    sub.add_parser("guard-bash", help="deny force-push and pushes to protected branches")
    scope = sub.add_parser("scope-write", help="allow writes only under the given directories")
    scope.add_argument("prefixes", nargs="+")
    read = sub.add_parser("deny-read", help="deny paths matching globs or policy categories")
    read.add_argument("targets", nargs="+")
    only = sub.add_parser("allow-only", help="allow only the given exact Bash commands")
    only.add_argument("commands", nargs="+")
    sub.add_parser("claims", help="Stop hook: verify run-tagged numbers against the registry")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    env = os.environ if environ is None else environ
    err = stderr if stderr is not None else sys.stderr
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse: usage error (2) or --help (0)
        if exc.code in (0, None):
            return 0
        fail_open = env.get(FAIL_OPEN_ENV, "") == "1"
        label = "WARNING" if fail_open else "BLOCKED"
        err.write(f"{label}: qcal hook arguments are invalid: {list(argv or sys.argv[1:])}\n")
        return 0 if fail_open else BLOCK_EXIT_CODE
    text = (stdin if stdin is not None else sys.stdin).read()
    fail_closed = args.hook in GUARD_HOOKS

    try:
        payload = HookPayload.parse(text)
        config = _load(payload, env)
        configure_logging(config, level=None, stream=err, environ=_quiet_env(config, env))
        mode = resolve_mode(config, env)
    except Exception as exc:  # noqa: BLE001 - any setup failure must fail closed for guards
        return _setup_failure(args.hook, exc, err, fail_closed=fail_closed)

    try:
        with gitutil.timeout_scope(config.float_value("git.timeout_s")):
            decision = _dispatch(args, payload, config, env, err)
    except Exception as exc:  # noqa: BLE001 - a crashing guard blocks; it never allows
        _log.debug("hook %s raised", args.hook, exc_info=True)
        if not fail_closed:
            err.write(f"WARNING: qcal {args.hook} hook error (not blocking): {exc}\n")
            return 0
        decision = Decision.denied(f"qcal {args.hook} guard error: {exc}", "internal_error")
    return emit(
        decision,
        hook=args.hook,
        mode=mode,
        config=config,
        stderr=err,
        environ=env,
        payload=payload.raw,
    )


def _dispatch(
    args: argparse.Namespace,
    payload: HookPayload,
    config: Config,
    env: Mapping[str, str],
    err: IO[str],
) -> Decision:
    policy = Policy.from_config(config)
    root = guards.project_root(config, env)
    handlers: dict[str, Callable[[], Decision]] = {
        "guard-paths": lambda: guards.guard_paths(payload, config=config, policy=policy, root=root),
        "guard-bash": lambda: guards.guard_bash(payload, config=config, policy=policy),
        "scope-write": lambda: guards.scope_write(
            payload, prefixes=args.prefixes, config=config, root=root
        ),
        "deny-read": lambda: guards.deny_read(
            payload, targets=args.targets, config=config, policy=policy, root=root
        ),
        "allow-only": lambda: guards.allow_only(payload, commands=args.commands),
        "claims": lambda: _claims(payload, config, err),
    }
    return handlers[args.hook]()


def _claims(payload: HookPayload, config: Config, err: IO[str]) -> Decision:
    from qcal.integrity.claims import check_claims, format_findings

    findings = check_claims(config)
    if not findings:
        return Decision.allowed("claims.ok")
    message = format_findings(findings, config.root)
    if payload.stop_hook_active:
        err.write(f"WARNING: claims check still failing; not blocking again.\n{message}\n")
        return Decision.allowed("claims.reentry")
    return Decision.denied(
        f"numbers without a verifiable run reference:\n{message}\n"
        "Regenerate tables with `qcal registry tables` or tag values with run references.",
        "claims.findings",
    )


def _load(payload: HookPayload, env: Mapping[str, str]) -> Config:
    # The variable naming the project directory is itself configurable; read it from
    # the defaults+environment layers before the repository root is known.
    bootstrap = load_config(Path.cwd(), environ=env, use_repo_file=False)
    root_hint = env.get(bootstrap.str_value("hooks.project_dir_env")) or payload.cwd
    root = find_root(Path(root_hint), environ=env) if root_hint else None
    return load_config(root, environ=env)


def _quiet_env(config: Config, env: Mapping[str, str]) -> Mapping[str, str]:
    """Hooks log at WARNING unless debugging, so stderr stays readable for Claude."""
    level_env = config.str_value("logging.level_env")
    if level_env in env or env.get(config.str_value("logging.debug_env")):
        return env
    return {**env, level_env: "WARNING"}


def _setup_failure(hook: str, exc: Exception, err: IO[str], *, fail_closed: bool) -> int:
    detail = str(exc) if isinstance(exc, PayloadError) else f"{type(exc).__name__}: {exc}"
    if fail_closed:
        err.write(f"BLOCKED: qcal {hook} guard could not start ({detail}); failing closed\n")
        return BLOCK_EXIT_CODE
    err.write(f"WARNING: qcal {hook} hook could not start ({detail}); not blocking\n")
    return 0


__all__ = ["FAIL_OPEN_ENV", "GUARD_HOOKS", "STOP_HOOKS", "Mode", "build_parser", "main"]
