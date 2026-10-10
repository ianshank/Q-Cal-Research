"""Integrity checks: ``init``, ``claims``, ``leakage``, ``licenses``, ``agent-layer``."""

from __future__ import annotations

import argparse
from typing import TextIO

from qcal.cli.common import EXIT_FAILED, EXIT_OK, SubParsers, add_json_flag, emit, emit_report
from qcal.config import Config


def cmd_init(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.initdocs import init_documents

    actions = init_documents(config, force=args.force, dry_run=args.dry_run)
    text = "\n".join(f"{a.action:>15}  {a.path}" for a in actions)
    emit(out, [a.__dict__ for a in actions], as_json=args.json, text=text)
    return EXIT_OK


def cmd_claims(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.integrity.claims import check_claims, format_findings

    findings = check_claims(config)
    payload = [
        {
            "path": str(f.path.relative_to(config.root)),
            "line": f.line,
            "kind": f.kind,
            "message": f.message,
        }
        for f in findings
    ]
    text = format_findings(findings, config.root) or "claims: PASS"
    emit(out, payload, as_json=args.json, text=text)
    return EXIT_FAILED if findings else EXIT_OK


def cmd_leakage(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.integrity.leakage import check_leakage

    report = check_leakage(config)
    data = report.to_dict()
    if args.if_present and not report.present:
        data["verdict"] = "SKIP"
        emit(out, data, as_json=args.json, text="leakage: SKIP (no split manifests yet)")
        return EXIT_OK
    return emit_report(out, report, args)


def cmd_licenses(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.integrity.licenses import check_licenses

    return emit_report(out, check_licenses(config, packages=not args.no_packages), args)


def cmd_agent_layer(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.integrity.agent_layer import check_agent_layer

    return emit_report(out, check_agent_layer(config), args)


def add_commands(sub: SubParsers) -> None:
    init = sub.add_parser("init", help="create Ian's document templates if missing")
    init.add_argument("--force", action="store_true")
    init.add_argument("--dry-run", action="store_true")
    add_json_flag(init)
    init.set_defaults(handler=cmd_init)

    claims = sub.add_parser("claims", help="verify run-tagged numbers in paper and claim files")
    add_json_flag(claims)
    claims.set_defaults(handler=cmd_claims)

    leak = sub.add_parser("leakage", help="split manifest disjointness and hashes")
    leak.add_argument("--if-present", action="store_true", help="pass when no manifests exist yet")
    add_json_flag(leak)
    leak.set_defaults(handler=cmd_leakage)

    lic = sub.add_parser("licenses", help="dependency, import and dataset-card license audit")
    lic.add_argument("--no-packages", action="store_true", help="skip installed-package checks")
    add_json_flag(lic)
    lic.set_defaults(handler=cmd_licenses)

    agents = sub.add_parser("agent-layer", help="validate .claude agents, skills, hooks, .mcp.json")
    add_json_flag(agents)
    agents.set_defaults(handler=cmd_agent_layer)
