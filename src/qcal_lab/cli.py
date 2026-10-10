"""``python -m qcal_lab``: the experiment program, the smoke test, status and split tools.

Exit codes follow ``qcal``: 0 success, 1 a run or check failed, 2 a usage or configuration
error. ``run`` is what ``executor.command`` invokes; a non-zero exit becomes a failed run
record with this program's log attached.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from qcal.config import ConfigError, find_root, load_config
from qcal.integrity.leakage import check_leakage, read_manifest
from qcal.log import configure_logging, get_logger
from qcal.registry.experiments import ExperimentsError
from qcal.reports import verdict
from qcal_lab.calib.base import CalibrationError
from qcal_lab.config import load_lab_config
from qcal_lab.data.coco import DatasetError, load_coco
from qcal_lab.data.splits import SplitError, manifest_path, partition, split_digest, write_manifest
from qcal_lab.evaluation import EvaluationError
from qcal_lab.experiment import PlanError, RunRequest, run_experiment, write_result
from qcal_lab.models import DetectorError
from qcal_lab.predictions import PredictionsError

_log = get_logger("lab.cli")

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
RUN_FAILURES = (
    CalibrationError,
    DatasetError,
    DetectorError,
    EvaluationError,
    ExperimentsError,
    PlanError,
    PredictionsError,
    SplitError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qcal_lab", description=__doc__)
    parser.add_argument("--root", type=Path, help="repository root (default: discovered)")
    parser.add_argument("--debug", action="store_true", help="DEBUG logs and tracebacks")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one pre-registered cell (called by the registry)")
    run.add_argument("--run-id", required=True)
    run.add_argument("--cell-id", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--result-path", type=Path, required=True)

    smoke = sub.add_parser("smoke", help="the whole loop on a synthetic fixture (< 5 min)")
    smoke.add_argument("--workdir", type=Path, help="keep the smoke project here")
    smoke.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="what still blocks a registered Phase 1 run")
    status.add_argument("--json", action="store_true")
    status.add_argument("--strict", action="store_true", help="exit 1 unless ready")

    splits = sub.add_parser("splits", help="create or verify split manifests")
    splits_sub = splits.add_subparsers(dest="splits_command", required=True)
    imp = splits_sub.add_parser("import", help="one split from every image of an annotation file")
    imp.add_argument("--annotations", type=Path, required=True)
    imp.add_argument("--split", required=True)
    part = splits_sub.add_parser("partition", help="disjoint seeded splits of one annotation file")
    part.add_argument("--annotations", type=Path, required=True)
    part.add_argument("--seed", type=int, required=True)
    part.add_argument("--sizes", required=True, help="split=count,... filled in this order")
    verify = splits_sub.add_parser(
        "verify", help="every manifest id exists in the annotations, and the splits are disjoint"
    )
    verify.add_argument("--annotations", type=Path, required=True)
    return parser


def _parse_sizes(text: str) -> list[tuple[str, int]]:
    sizes: list[tuple[str, int]] = []
    for item in filter(None, (part.strip() for part in text.split(","))):
        name, sep, count = item.partition("=")
        if not sep or not name or not count.isdigit():
            raise SplitError(f"--sizes entries look like split=count, got {item!r}")
        sizes.append((name, int(count)))
    return sizes


def _cmd_run(args: argparse.Namespace, root: Path) -> int:
    request = RunRequest(root, args.run_id, args.cell_id, args.seed, args.result_path)
    try:
        result = run_experiment(request)
    except RUN_FAILURES as exc:
        _log.error("run %s failed: %s", args.run_id, exc)
        return EXIT_FAILED
    write_result(args.result_path, result)
    _log.info(
        "run %s: %d metric(s), %d artifact(s)",
        args.run_id,
        len(result.metrics),
        len(result.artifacts),
    )
    return EXIT_OK


def _cmd_smoke(args: argparse.Namespace, root: Path, out: IO[str]) -> int:
    from qcal_lab.smoke import run_smoke

    lab = load_lab_config(root)
    if args.workdir is not None:
        report = run_smoke(lab, args.workdir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="qcal-smoke-") as tmp:
            report = run_smoke(lab, Path(tmp))
    out.write(
        (json.dumps(report.to_dict(), indent=2) if args.json else report.render_text()) + "\n"
    )
    return EXIT_OK if report.passed else EXIT_FAILED


def _cmd_status(args: argparse.Namespace, root: Path, out: IO[str]) -> int:
    from qcal_lab.status import build_status

    report = build_status(load_config(root), load_lab_config(root))
    out.write(
        (json.dumps(report.to_dict(), indent=2) if args.json else report.render_text()) + "\n"
    )
    return EXIT_FAILED if args.strict and not report.passed else EXIT_OK


def _cmd_splits(args: argparse.Namespace, root: Path, out: IO[str]) -> int:
    config = load_config(root)
    dataset = load_coco(args.annotations)
    comment = config.str_value("data.comment_prefix")
    source = {"source": args.annotations.name, "source_sha256": dataset.sha256}
    if args.splits_command == "import":
        path = manifest_path(config, args.split)
        digest = write_manifest(path, dataset.image_ids(), source, comment)
        out.write(f"{args.split}: {len(dataset.images)} ids, sha256 {digest}\n")
        return EXIT_OK
    if args.splits_command == "partition":
        parts = partition(dataset.image_ids(), _parse_sizes(args.sizes), args.seed)
        header = {**source, "seed": str(args.seed), "sizes": args.sizes}
        for split, ids in parts.items():
            digest = write_manifest(manifest_path(config, split), ids, header, comment)
            out.write(f"{split}: {len(ids)} ids, sha256 {digest}\n")
        return EXIT_OK
    failed = False
    for split in config.str_list("data.splits"):
        path = manifest_path(config, split)
        if not path.is_file():
            out.write(f"{split}: no manifest\n")
            continue
        listed = read_manifest(path.read_text("utf-8"), comment)
        missing = [i for i in listed if i not in dataset.images]
        failed |= bool(missing)
        state = f"{len(missing)} id(s) not in {args.annotations.name}" if missing else "ok"
        out.write(f"{split}: {len(listed)} ids, sha256 {split_digest(listed)}, {state}\n")
    # The verdict is check_leakage's: a missing manifest or a duplicated id fails, as an overlap.
    leakage = check_leakage(config)
    for split, count in leakage.duplicates.items():
        out.write(f"duplicate ids in {split}: {count}\n")
    for pair, shared in leakage.overlaps.items():
        out.write(f"overlap {pair}: {len(shared)} id(s), e.g. {', '.join(shared[:3])}\n")
    out.write(f"leakage: {verdict(leakage.passed)}\n")
    return EXIT_FAILED if failed or not leakage.passed else EXIT_OK


def main(argv: Sequence[str] | None = None, *, out: IO[str] | None = None) -> int:
    stream = out or sys.stdout
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    configure_logging(None, level="DEBUG" if args.debug else None)
    root = (args.root or find_root()).resolve()
    try:
        if args.command == "run":
            return _cmd_run(args, root)
        return _COMMANDS[args.command](args, root, stream)
    except ConfigError as exc:
        _log.error("%s", exc)
        return EXIT_USAGE
    except RUN_FAILURES as exc:
        _log.error("%s", exc)
        return EXIT_FAILED


_COMMANDS = {"smoke": _cmd_smoke, "status": _cmd_status, "splits": _cmd_splits}


__all__ = ["EXIT_FAILED", "EXIT_OK", "EXIT_USAGE", "build_parser", "main"]
