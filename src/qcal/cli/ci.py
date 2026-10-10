"""``qcal ci ...``: checks that read only git objects (safe under pull_request_target)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TextIO

from qcal.cli.common import EXIT_FAILED, EXIT_OK, SubParsers, add_json_flag, emit
from qcal.config import Config


def _write_json(path: str | None, data: Any) -> None:
    if path:
        Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def cmd_verify_signatures(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.ci.signatures import github_annotations, verify_signatures

    report = verify_signatures(
        config.root, args.base, args.head, policy_ref=args.policy_ref, mode=args.mode
    )
    data = report.to_dict()
    _write_json(args.json_out, data)
    if args.github:
        out.writelines(line + "\n" for line in github_annotations(report))
    emit(out, data, as_json=args.json, text=report.render_text())
    return EXIT_OK if report.passed else EXIT_FAILED


def cmd_registry_immutable(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.ci.immutability import check_registry_immutable

    report = check_registry_immutable(config.root, args.base, args.head, policy_ref=args.policy_ref)
    data = report.to_dict()
    _write_json(args.json_out, data)
    if args.github:
        out.writelines(f"::error::registry: {v}\n" for v in report.violations)
    emit(out, data, as_json=args.json, text=report.render_text())
    return EXIT_OK if report.passed else EXIT_FAILED


def cmd_review_check(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.ci.review import check_review, github_annotations

    report = check_review(
        config.root,
        args.base,
        args.head,
        args.branch,
        policy_ref=args.policy_ref,
        mode=args.mode,
    )
    data = report.to_dict()
    _write_json(args.json_out, data)
    if args.github:
        out.writelines(line + "\n" for line in github_annotations(report))
    emit(out, data, as_json=args.json, text=report.render_text())
    return EXIT_OK if report.passed else EXIT_FAILED


def add_commands(sub: SubParsers) -> None:
    ci = sub.add_parser("ci", help="CI integrity checks (git objects only)")
    commands = ci.add_subparsers(dest="ci_command", required=True)
    sig = commands.add_parser("verify-signatures", help="protected-path commits must be signed")
    imm = commands.add_parser("registry-immutable", help="run records may only be added")
    rev = commands.add_parser(
        "review-check", help="the other model's current, approving review is committed"
    )
    for parser in (sig, imm, rev):
        parser.add_argument("--base", required=True)
        parser.add_argument("--head", required=True)
        parser.add_argument(
            "--policy-ref", help="ref holding qcal.toml and allowed_signers (default: base)"
        )
        parser.add_argument("--json-out", help="also write the JSON report to this file")
        parser.add_argument("--github", action="store_true", help="print Actions annotations")
        add_json_flag(parser)
    for parser, key in ((sig, "signing.mode"), (rev, "review.mode")):
        parser.add_argument(
            "--mode", choices=["bootstrap", "enforce"], help=f"override {key} locally"
        )
    rev.add_argument("--branch", required=True, help="the pull request's head branch name")
    sig.set_defaults(handler=cmd_verify_signatures)
    imm.set_defaults(handler=cmd_registry_immutable)
    rev.set_defaults(handler=cmd_review_check)
