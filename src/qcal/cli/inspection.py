"""Debugging aids: ``qcal policy list|check`` and ``qcal config``."""

from __future__ import annotations

import argparse
from typing import TextIO

from qcal.cli.common import EXIT_FAILED, EXIT_OK, SubParsers, add_json_flag, emit
from qcal.config import Config, ConfigError, config_key_problems


def cmd_policy_list(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal import gitutil
    from qcal.globs import iter_files
    from qcal.policy import Policy

    policy = Policy.from_config(config)
    unknown = [c for c in args.category if c not in policy.category_names]
    if unknown:
        known = list(policy.category_names)
        raise ConfigError(f"unknown policy categories {unknown}; known: {known}")
    tracked = gitutil.try_git(["ls-files", "-z"], config.root)
    if tracked is not None:
        candidates = [path for path in tracked.split("\0") if path]
    else:
        patterns = [p for c in args.category for p in policy.patterns(c)]
        candidates = [
            p.relative_to(config.root).as_posix() for p in iter_files(config.root, patterns)
        ]
    matches = sorted(f for f in candidates if policy.in_categories(f, args.category))
    emit(out, matches, as_json=args.json, text="\n".join(matches))
    return EXIT_OK


def cmd_policy_check(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.policy import Policy

    policy = Policy.from_config(config)
    verdicts = [policy.evaluate(p, config.root) for p in args.paths]
    payload = [
        {
            "path": v.raw,
            "relative": v.relative,
            "categories": list(v.categories),
            "clean_room_hits": list(v.clean_room_hits),
        }
        for v in verdicts
    ]
    text = "\n".join(
        f"{v.raw}: {', '.join(v.categories) or 'unprotected'}"
        + (f" [clean-room: {', '.join(v.clean_room_hits)}]" if v.clean_room_hits else "")
        for v in verdicts
    )
    emit(out, payload, as_json=args.json, text=text)
    return EXIT_OK


def cmd_config(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    if args.check:
        from qcal.reports import verdict

        problems = config_key_problems(config.data)
        report = {"verdict": verdict(not problems), "problems": problems}
        text = "\n".join([f"config: {verdict(not problems)}", *(f"  {p}" for p in problems)])
        emit(out, report, as_json=args.json, text=text)
        return EXIT_FAILED if problems else EXIT_OK
    payload: dict[str, object] = {
        "root": str(config.root),
        "sources": list(config.sources),
        "environment_inputs": dict(config.environment_inputs),
        "config": config.data,
    }
    emit(out, payload, as_json=True, text="")
    return EXIT_OK


def add_commands(sub: SubParsers) -> None:
    pol = sub.add_parser("policy", help="inspect the path policy")
    commands = pol.add_subparsers(dest="policy_command", required=True)
    pol_list = commands.add_parser("list", help="tracked files in the given categories")
    pol_list.add_argument("--category", action="append", required=True)
    add_json_flag(pol_list)
    pol_list.set_defaults(handler=cmd_policy_list)
    pol_check = commands.add_parser("check", help="classify paths (debugging aid)")
    pol_check.add_argument("paths", nargs="+")
    add_json_flag(pol_check)
    pol_check.set_defaults(handler=cmd_policy_check)

    cfg = sub.add_parser("config", help="print the merged configuration and its sources")
    cfg.add_argument(
        "--check",
        action="store_true",
        help="fail on unknown, removed or mistyped keys (registered runs refuse them)",
    )
    add_json_flag(cfg)
    cfg.set_defaults(handler=cmd_config)
