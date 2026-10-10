"""``qcal registry ...``: run, run-batch, index, audit, tables, cells."""

from __future__ import annotations

import argparse
from typing import Any, TextIO

from qcal.cli.common import (
    EXIT_FAILED,
    EXIT_OK,
    SubParsers,
    add_experiments_arg,
    add_json_flag,
    emit,
    load_experiments,
    seed_list,
)
from qcal.config import Config
from qcal.log import get_logger

_log = get_logger("cli.registry")


def _store(config: Config) -> Any:
    from qcal.registry.store import RegistryStore

    return RegistryStore(config.path("registry_dir"))


def cmd_run(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.executor import build_executor
    from qcal.registry.runner import Runner

    runner = Runner(
        config,
        load_experiments(config, args.experiments),
        _store(config),
        build_executor(config, args.executor or config.str_value("executor.kind")),
    )
    record = runner.run(args.cell, args.seed, supersedes=args.supersedes, reason=args.reason)
    emit(out, record.to_dict(), as_json=args.json, text=f"{record.run_id}: {record.status}")
    return EXIT_OK if record.status == config.str_value("registry.ok_status") else EXIT_FAILED


def cmd_run_batch(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.executor import build_executor
    from qcal.registry.runner import Runner

    kind = args.executor or config.str_value("executor.kind")
    executor: Any = None if args.dry_run else build_executor(config, kind)
    runner = Runner(config, load_experiments(config, args.experiments), _store(config), executor)
    batch = runner.run_batch(
        args.pattern,
        seeds=args.seeds,
        max_runs=args.max_runs,
        keep_going=args.keep_going,
        rerun=args.rerun,
        dry_run=args.dry_run,
        reason=args.reason,
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
    emit(out, payload, as_json=args.json, text=text)
    return EXIT_OK if batch.ok else EXIT_FAILED


def cmd_index(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.index import write_index

    result = write_index(config, _store(config), check=args.check)
    if args.check:
        state = "stale" if result.changed else "up to date"
    else:
        state = "updated" if result.changed else "unchanged"
    emit(
        out,
        {"path": str(result.path), "rows": result.rows, "changed": result.changed},
        as_json=args.json,
        text=f"{result.path.relative_to(config.root)}: {result.rows} rows, {state}",
    )
    return EXIT_FAILED if args.check and result.changed else EXIT_OK


def cmd_audit(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.audit import audit

    report = audit(config, load_experiments(config, args.experiments), _store(config).load_all())
    emit(out, report.to_dict(), as_json=args.json, text=report.render_text())
    return EXIT_OK if report.ok(strict=args.strict) else EXIT_FAILED


def cmd_tables(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    from qcal.registry.tables import build_tables

    results = build_tables(config, check=args.check)
    rows = [{"path": str(r.path.relative_to(config.root)), "changed": r.changed} for r in results]
    text = "\n".join(
        f"{'stale' if r['changed'] and args.check else 'ok':>6}  {r['path']}" for r in rows
    )
    emit(out, rows, as_json=args.json, text=text or "no table specs found")
    return EXIT_FAILED if args.check and any(r.changed for r in results) else EXIT_OK


def cmd_cells(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    import yaml

    from qcal.registry.cells import cells_to_yaml_entries, expand_design, summarize
    from qcal.registry.experiments import ExperimentsError

    experiments = load_experiments(config, args.experiments)
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
    text = f"{len(cells)} cells\n" + "\n".join(f"  {k}: {v}" for k, v in factors.items())
    emit(out, {"cells": len(cells), "factors": factors}, as_json=args.json, text=text)
    return EXIT_OK


def add_commands(sub: SubParsers) -> None:
    registry = sub.add_parser("registry", help="run registry commands")
    commands = registry.add_subparsers(dest="registry_command", required=True)

    run = commands.add_parser("run", help="run one pre-registered cell and seed")
    run.add_argument("cell")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--executor", help="executor kind (default: executor.kind)")
    run.add_argument(
        "--supersedes", metavar="RUN_ID", help="replace this current run of the same cell and seed"
    )
    run.add_argument("--reason", default="", help="why the run is replaced (recorded)")
    add_experiments_arg(run)
    add_json_flag(run)
    run.set_defaults(handler=cmd_run)

    batch = commands.add_parser(
        "run-batch", help="run every pre-registered cell matching a pattern"
    )
    batch.add_argument("pattern", help="fnmatch pattern(s) over cell ids, comma-separated")
    batch.add_argument(
        "--seeds", type=seed_list, help="comma-separated subset of pre-registered seeds"
    )
    batch.add_argument("--max-runs", type=int)
    batch.add_argument("--keep-going", action="store_true")
    batch.add_argument(
        "--rerun", action="store_true", help="re-run completed pairs; new records supersede old"
    )
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--reason", default="", help="why --rerun replaces runs (recorded)")
    batch.add_argument("--executor", help="executor kind (default: executor.kind)")
    add_experiments_arg(batch)
    add_json_flag(batch)
    batch.set_defaults(handler=cmd_run_batch)

    index = commands.add_parser("index", help="regenerate runs/index.csv from run records")
    index.add_argument("--check", action="store_true", help="fail if the committed index is stale")
    add_json_flag(index)
    index.set_defaults(handler=cmd_index)

    aud = commands.add_parser("audit", help="pre-registered coverage versus the registry")
    aud.add_argument(
        "--strict",
        action="store_true",
        help="also fail on missing or failed-only pairs and placeholders",
    )
    add_experiments_arg(aud)
    add_json_flag(aud)
    aud.set_defaults(handler=cmd_audit)

    tab = commands.add_parser("tables", help="build LaTeX tables from the index")
    tab.add_argument("--check", action="store_true", help="fail if committed tables are stale")
    add_json_flag(tab)
    tab.set_defaults(handler=cmd_tables)

    cells = commands.add_parser(
        "cells", help="expand the design section into an explicit cell list"
    )
    cells.add_argument("--emit", action="store_true", help="print YAML for Ian to paste into cells")
    add_experiments_arg(cells)
    add_json_flag(cells)
    cells.set_defaults(handler=cmd_cells)
