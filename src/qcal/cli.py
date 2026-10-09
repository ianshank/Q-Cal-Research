"""Command-line interface: ``qcal <command>`` plus the plan's ``qcal-registry`` alias.

Exit codes: 0 success, 1 a check failed or a run failed, 2 usage or
configuration error. ``--debug`` (or ``$QCAL_DEBUG``) prints tracebacks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, TextIO

from qcal import __version__
from qcal.config import Config, ConfigError, load_config
from qcal.log import configure_logging, get_logger, is_truthy

_log = get_logger("cli")

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_USAGE: Final = 2

Handler = Callable[[argparse.Namespace, Config, TextIO], int]


def _emit(out: TextIO, payload: Any, as_json: bool, text: str) -> None:
    out.write(
        (json.dumps(payload, indent=2, sort_keys=True, default=str) if as_json else text) + "\n"
    )


# --------------------------------------------------------------------------- handlers
def _cmd_init(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.initdocs import init_documents

    actions = init_documents(config, force=args.force, dry_run=args.dry_run)
    _emit(
        out,
        [a.__dict__ for a in actions],
        args.json,
        "\n".join(f"{a.action:>12}  {a.path}" for a in actions),
    )
    return EXIT_OK


def _load_experiments(config: Config, path: str | None) -> Any:
    from qcal.registry.experiments import load_experiments

    return load_experiments(config, Path(path) if path else None)


def _cmd_run(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.executor import build_executor
    from qcal.registry.runner import Runner
    from qcal.registry.store import RegistryStore

    runner = Runner(
        config,
        _load_experiments(config, args.experiments),
        RegistryStore(config.path("registry_dir")),
        build_executor(config, args.executor),
    )
    record = runner.run(args.cell, args.seed)
    _emit(out, record.to_dict(), args.json, f"{record.run_id}: {record.status}")
    return EXIT_OK if record.status == config.str_value("registry.ok_status") else EXIT_FAILED


def _cmd_run_batch(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.executor import build_executor
    from qcal.registry.runner import Runner
    from qcal.registry.store import RegistryStore

    experiments = _load_experiments(config, args.experiments)
    store = RegistryStore(config.path("registry_dir"))
    executor: Any = None if args.dry_run else build_executor(config, args.executor)
    runner = Runner(config, experiments, store, executor)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    batch = runner.run_batch(
        args.pattern,
        seeds=seeds,
        max_runs=args.max_runs,
        keep_going=args.keep_going,
        rerun=args.rerun,
        dry_run=args.dry_run,
    )
    payload = {
        "planned": [f"{p.cell.id}@{p.seed}" for p in batch.planned],
        "completed": [r.run_id for r in batch.completed],
        "failed": [r.run_id for r in batch.failed],
        "skipped": [f"{c}@{s}: {why}" for c, s, why in batch.skipped],
    }
    text = "\n".join(
        f"{k}: {len(v)}" + "".join(f"\n  - {x}" for x in v) for k, v in payload.items()
    )
    _emit(out, payload, args.json, text)
    return EXIT_OK if batch.ok else EXIT_FAILED


def _cmd_index(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.index import write_index
    from qcal.registry.store import RegistryStore

    result = write_index(config, RegistryStore(config.path("registry_dir")), check=args.check)
    state = (
        ("stale" if result.changed else "up to date")
        if args.check
        else ("updated" if result.changed else "unchanged")
    )
    _emit(
        out,
        {"path": str(result.path), "rows": result.rows, "changed": result.changed},
        args.json,
        f"{result.path.relative_to(config.root)}: {result.rows} rows, {state}",
    )
    return EXIT_FAILED if args.check and result.changed else EXIT_OK


def _cmd_audit(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.audit import audit
    from qcal.registry.store import RegistryStore

    report = audit(
        config,
        _load_experiments(config, args.experiments),
        RegistryStore(config.path("registry_dir")).load_all(),
    )
    _emit(out, report.to_dict(), args.json, report.render_text())
    return EXIT_OK if report.ok(strict=args.strict) else EXIT_FAILED


def _cmd_tables(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.tables import build_tables

    results = build_tables(config, check=args.check)
    rows = [{"path": str(r.path.relative_to(config.root)), "changed": r.changed} for r in results]
    text = "\n".join(
        f"{'stale' if r['changed'] and args.check else 'ok':>6}  {r['path']}" for r in rows
    )
    _emit(out, rows, args.json, text or "no table specs found")
    return EXIT_FAILED if args.check and any(r.changed for r in results) else EXIT_OK


def _cmd_cells(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    import yaml

    from qcal.registry.cells import cells_to_yaml_entries, expand_design, summarize
    from qcal.registry.experiments import ExperimentsError

    experiments = _load_experiments(config, args.experiments)
    design = experiments.data.get(config.str_value("experiments.design_key"))
    if not design:
        raise ExperimentsError("EXPERIMENTS.yaml has no design section to expand")
    cells = expand_design(
        design,
        prefix=config.str_value("experiments.cell_id_prefix"),
        length=config.int_value("experiments.cell_id_hash_length"),
    )
    threshold = config.int_value("experiments.max_cells_warning")
    if len(cells) > threshold:
        _log.warning(
            "design expands to %d cells, above the %d warning threshold", len(cells), threshold
        )
    if args.emit:
        entries = cells_to_yaml_entries(cells, config.str_value("experiments.cell_id_key"))
        out.write(
            yaml.safe_dump({config.str_value("experiments.cells_key"): entries}, sort_keys=False)
        )
        return EXIT_OK
    factors = summarize(cells)
    summary = {"cells": len(cells), "factors": factors}
    text = f"{len(cells)} cells\n" + "\n".join(f"  {k}: {v}" for k, v in factors.items())
    _emit(out, summary, args.json, text)
    return EXIT_OK


def _cmd_claims(args: argparse.Namespace, config: Config, out: TextIO) -> int:
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
    _emit(out, payload, args.json, format_findings(findings, config.root) or "claims: PASS")
    return EXIT_FAILED if findings else EXIT_OK


def _cmd_leakage(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.integrity.leakage import check_leakage

    report = check_leakage(config)
    data = report.to_dict()
    if args.if_present and not report.present:
        data["verdict"] = "SKIP"
        _emit(out, data, args.json, "leakage: SKIP (no split manifests yet)")
        return EXIT_OK
    _emit(out, data, args.json, json.dumps(data, indent=2))
    return EXIT_OK if report.passed else EXIT_FAILED


def _cmd_licenses(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.integrity.licenses import check_licenses

    report = check_licenses(config, packages=not args.no_packages)
    data = report.to_dict()
    text = "\n".join(
        [
            f"licenses: {data['verdict']}",
            *(f"  error: {e}" for e in report.errors),
            *(f"  warning: {w}" for w in report.warnings),
        ]
    )
    _emit(out, data, args.json, text)
    return EXIT_OK if report.passed else EXIT_FAILED


def _cmd_agent_layer(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.integrity.agent_layer import check_agent_layer

    report = check_agent_layer(config)
    text = "\n".join(
        [
            f"agent layer: {'PASS' if report.passed else 'FAIL'} {report.checked}",
            *(f"  error: {e}" for e in report.errors),
        ]
    )
    _emit(out, report.to_dict(), args.json, text)
    return EXIT_OK if report.passed else EXIT_FAILED


def _cmd_verify_signatures(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.ci.signatures import github_annotations, verify_signatures

    report = verify_signatures(
        config.root, args.base, args.head, policy_ref=args.policy_ref, mode=args.mode
    )
    data = report.to_dict()
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.github:
        for line in github_annotations(report):
            out.write(line + "\n")
    _emit(out, data, args.json, json.dumps(data, indent=2))
    return EXIT_OK if report.passed else EXIT_FAILED


def _cmd_registry_immutable(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.ci.immutability import check_registry_immutable

    report = check_registry_immutable(config.root, args.base, args.head, policy_ref=args.policy_ref)
    _emit(out, report.to_dict(), args.json, json.dumps(report.to_dict(), indent=2))
    return EXIT_OK if report.passed else EXIT_FAILED


def _cmd_policy_list(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal import gitutil
    from qcal.globs import iter_files
    from qcal.policy import Policy

    policy = Policy.from_config(config)
    unknown = [c for c in args.category if c not in policy.category_names]
    if unknown:
        raise ConfigError(
            f"unknown policy categories {unknown}; known: {list(policy.category_names)}"
        )
    tracked = gitutil.try_git(["ls-files"], config.root)
    if tracked is not None:
        candidates = [line for line in tracked.splitlines() if line]
    else:
        patterns = [p for c in args.category for p in policy.patterns(c)]
        candidates = [
            p.relative_to(config.root).as_posix() for p in iter_files(config.root, patterns)
        ]
    matches = sorted(f for f in candidates if policy.in_categories(f, args.category))
    _emit(out, matches, args.json, "\n".join(matches))
    return EXIT_OK


def _cmd_policy_check(args: argparse.Namespace, config: Config, out: TextIO) -> int:
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
    _emit(out, payload, args.json, text)
    return EXIT_OK


def _cmd_config(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    payload = {"root": str(config.root), "sources": list(config.sources), "config": config.data}
    _emit(out, payload, True, "")
    return EXIT_OK


# --------------------------------------------------------------------------- parser
def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def _experiments_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiments", help="path to EXPERIMENTS.yaml (default from config)")


def _add_registry(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = sub.add_parser("run", help="run one pre-registered cell and seed")
    run.add_argument("cell")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--executor", default="subprocess")
    _experiments_arg(run)
    _common(run)
    run.set_defaults(handler=_cmd_run)

    batch = sub.add_parser("run-batch", help="run every pre-registered cell matching a pattern")
    batch.add_argument("pattern", help="fnmatch pattern(s) over cell ids, comma-separated")
    batch.add_argument("--seeds", help="comma-separated subset of pre-registered seeds")
    batch.add_argument("--max-runs", type=int)
    batch.add_argument("--keep-going", action="store_true")
    batch.add_argument(
        "--rerun", action="store_true", help="re-run completed pairs; new records supersede old"
    )
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--executor", default="subprocess")
    _experiments_arg(batch)
    _common(batch)
    batch.set_defaults(handler=_cmd_run_batch)

    index = sub.add_parser("index", help="regenerate runs/index.csv from run records")
    index.add_argument("--check", action="store_true", help="fail if the committed index is stale")
    _common(index)
    index.set_defaults(handler=_cmd_index)

    aud = sub.add_parser("audit", help="pre-registered coverage versus the registry")
    aud.add_argument(
        "--strict", action="store_true", help="also fail on missing cells and placeholders"
    )
    _experiments_arg(aud)
    _common(aud)
    aud.set_defaults(handler=_cmd_audit)

    tab = sub.add_parser("tables", help="build LaTeX tables from the index")
    tab.add_argument("--check", action="store_true", help="fail if committed tables are stale")
    _common(tab)
    tab.set_defaults(handler=_cmd_tables)

    cells = sub.add_parser("cells", help="expand the design section into an explicit cell list")
    cells.add_argument("--emit", action="store_true", help="print YAML for Ian to paste into cells")
    _experiments_arg(cells)
    _common(cells)
    cells.set_defaults(handler=_cmd_cells)


def _add_checks(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    init = sub.add_parser("init", help="create Ian's document templates if missing")
    init.add_argument("--force", action="store_true")
    init.add_argument("--dry-run", action="store_true")
    _common(init)
    init.set_defaults(handler=_cmd_init)

    claims = sub.add_parser("claims", help="verify run-tagged numbers in paper and claim files")
    _common(claims)
    claims.set_defaults(handler=_cmd_claims)

    leak = sub.add_parser("leakage", help="split manifest disjointness and hashes")
    leak.add_argument("--if-present", action="store_true", help="pass when no manifests exist yet")
    _common(leak)
    leak.set_defaults(handler=_cmd_leakage)

    lic = sub.add_parser("licenses", help="dependency, import and dataset-card license audit")
    lic.add_argument("--no-packages", action="store_true", help="skip installed-package checks")
    _common(lic)
    lic.set_defaults(handler=_cmd_licenses)

    agents = sub.add_parser("agent-layer", help="validate .claude agents, skills, hooks, .mcp.json")
    _common(agents)
    agents.set_defaults(handler=_cmd_agent_layer)


def _add_ci(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ci = sub.add_parser("ci", help="CI integrity checks (git objects only)")
    ci_sub = ci.add_subparsers(dest="ci_command", required=True)
    sig = ci_sub.add_parser("verify-signatures", help="protected-path commits must be signed")
    imm = ci_sub.add_parser("registry-immutable", help="run records may only be added")
    for p in (sig, imm):
        p.add_argument("--base", required=True)
        p.add_argument("--head", required=True)
        p.add_argument(
            "--policy-ref", help="ref holding qcal.toml and allowed_signers (default: base)"
        )
        _common(p)
    sig.add_argument(
        "--mode", choices=["bootstrap", "enforce"], help="override signing.mode locally"
    )
    sig.add_argument("--json-out")
    sig.add_argument("--github", action="store_true", help="print GitHub Actions annotations")
    sig.set_defaults(handler=_cmd_verify_signatures)
    imm.set_defaults(handler=_cmd_registry_immutable)


def _add_policy(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    pol = sub.add_parser("policy", help="inspect the path policy")
    pol_sub = pol.add_subparsers(dest="policy_command", required=True)
    pol_list = pol_sub.add_parser("list", help="tracked files in the given categories")
    pol_list.add_argument("--category", action="append", required=True)
    _common(pol_list)
    pol_list.set_defaults(handler=_cmd_policy_list)
    pol_check = pol_sub.add_parser("check", help="classify paths (debugging aid)")
    pol_check.add_argument("paths", nargs="+")
    _common(pol_check)
    pol_check.set_defaults(handler=_cmd_policy_check)

    cfg = sub.add_parser("config", help="print the merged configuration and its sources")
    cfg.set_defaults(handler=_cmd_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qcal", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"qcal {__version__}")
    parser.add_argument("--root", help="repository root (default: discovered)")
    parser.add_argument("--log-level", help="override logging.level")
    parser.add_argument("--log-format", choices=["text", "json"])
    parser.add_argument("--debug", action="store_true", help="DEBUG logging and tracebacks")
    sub = parser.add_subparsers(dest="command", required=True)
    registry = sub.add_parser("registry", help="run registry commands")
    _add_registry(registry.add_subparsers(dest="registry_command", required=True))
    _add_checks(sub)
    _add_ci(sub)
    _add_policy(sub)
    hook = sub.add_parser(
        "hook", help="run a Claude Code hook (reads JSON on stdin)", add_help=False
    )
    hook.add_argument("hook_args", nargs=argparse.REMAINDER)
    hook.set_defaults(handler=None)
    return parser


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    stream = out if out is not None else sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "hook":
        from qcal.hooks.cli import main as hook_main

        return hook_main(args.hook_args)
    debug = args.debug
    try:
        config = load_config(Path(args.root) if args.root else None)
        debug = debug or is_truthy(_env(config.str_value("logging.debug_env")))
        configure_logging(
            config, level="DEBUG" if args.debug else args.log_level, fmt=args.log_format
        )
        _log.debug("config sources: %s", ", ".join(config.sources))
        handler: Handler = args.handler
        return handler(args, config, stream)
    except Exception as exc:
        if debug:
            raise
        message = f"error: {exc}"
        sys.stderr.write(message + "\n")
        return EXIT_USAGE if _is_usage_error(exc) else EXIT_FAILED


def _is_usage_error(exc: Exception) -> bool:
    from qcal.registry.experiments import ExperimentsError
    from qcal.registry.runner import RunRefusedError

    return isinstance(exc, ConfigError | ExperimentsError | RunRefusedError | FileNotFoundError)


def _env(name: str) -> str | None:
    import os

    return os.environ.get(name)


def registry_main(argv: Sequence[str] | None = None) -> int:
    return main(["registry", *(sys.argv[1:] if argv is None else argv)])


def claims_main(argv: Sequence[str] | None = None) -> int:
    return main(["claims", *(sys.argv[1:] if argv is None else argv)])
