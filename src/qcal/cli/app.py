"""Parser assembly and the top-level ``main``.

Command groups are plain functions ``add_commands(subparsers)`` listed in
:data:`COMMAND_GROUPS`; a new group is added by appending one, with no change to
``main``. Every handler has the signature ``(args, config, out) -> exit code``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, TextIO

from qcal import __version__, gitutil
from qcal.cli import checks, ci, inspection, registry
from qcal.cli.common import EXIT_FAILED, EXIT_USAGE, Handler, SubParsers
from qcal.config import ConfigError, load_config, load_defaults
from qcal.log import configure_logging, get_logger, is_truthy

_log = get_logger("cli")
_DESCRIPTION: Final = (
    "Q-Cal research integrity tooling. Exit codes: 0 success, 1 a check or run failed, "
    "2 usage or configuration error. --debug (or $QCAL_DEBUG) prints tracebacks."
)

COMMAND_GROUPS: list[Callable[[SubParsers], None]] = [
    registry.add_commands,
    checks.add_commands,
    ci.add_commands,
    inspection.add_commands,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qcal", description=_DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"qcal {__version__}")
    parser.add_argument("--root", help="repository root (default: discovered)")
    parser.add_argument("--log-level", help="override logging.level")
    parser.add_argument("--log-format", choices=["text", "json"])
    parser.add_argument("--debug", action="store_true", help="DEBUG logging and tracebacks")
    sub = parser.add_subparsers(dest="command", required=True)
    for add_group in COMMAND_GROUPS:
        add_group(sub)
    hook = sub.add_parser(
        "hook", help="run a Claude Code hook (reads JSON on stdin)", add_help=False
    )
    hook.add_argument("hook_args", nargs=argparse.REMAINDER)
    hook.set_defaults(handler=None)
    return parser


def usage_errors() -> tuple[type[Exception], ...]:
    """Exceptions that mean "the invocation or configuration is wrong" (exit 2)."""
    from qcal.ci.signatures import RefError
    from qcal.components import UnknownComponentError
    from qcal.registry.experiments import ExperimentsError
    from qcal.registry.runner import RunRefusedError

    return (
        ConfigError,
        ExperimentsError,
        RunRefusedError,
        FileNotFoundError,
        RefError,
        UnknownComponentError,
    )


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    stream = out if out is not None else sys.stdout
    args = build_parser().parse_args(argv)
    if args.command == "hook":
        from qcal.hooks.cli import main as hook_main

        return hook_main(args.hook_args)
    # The packaged debug variable is read first, so it also explains a config that fails to load.
    debug = args.debug or is_truthy(os.environ.get(load_defaults()["logging"]["debug_env"]))
    try:
        config = load_config(Path(args.root) if args.root else None)
        debug = debug or is_truthy(os.environ.get(config.str_value("logging.debug_env")))
        configure_logging(
            config, level="DEBUG" if args.debug else args.log_level, fmt=args.log_format
        )
        _log.debug("command=%s sources=%s", args.command, ", ".join(config.sources))
        handler: Handler = args.handler
        with gitutil.timeout_scope(config.float_value("git.timeout_s")):
            return handler(args, config, stream)
    except Exception as exc:  # the CLI boundary: report, never traceback unless debugging
        if debug:
            raise
        _log.debug("command failed", exc_info=True)
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_USAGE if isinstance(exc, usage_errors()) else EXIT_FAILED


def registry_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the plan's ``qcal-registry`` alias."""
    return main(["registry", *(sys.argv[1:] if argv is None else argv)])


def claims_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``qcal-claims`` (used by pre-commit)."""
    return main(["claims", *(sys.argv[1:] if argv is None else argv)])
