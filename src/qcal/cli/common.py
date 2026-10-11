"""Shared CLI plumbing: exit codes, output, common arguments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO, TypeAlias

from qcal.config import Config

if TYPE_CHECKING:
    from qcal.registry.experiments import Experiments
    from qcal.reports import CheckReport

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_USAGE: Final = 2
EXIT_INTERRUPTED: Final = 130  # 128 + SIGINT, as shells report it

Handler: TypeAlias = Callable[[argparse.Namespace, Config, TextIO], int]
# argparse exposes no public name for the object add_subparsers() returns.
SubParsers: TypeAlias = "argparse._SubParsersAction[argparse.ArgumentParser]"


def emit(out: TextIO, payload: Any, *, as_json: bool, text: str) -> None:
    """Write either the JSON payload or the human text, followed by a newline."""
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) if as_json else text
    out.write(rendered + "\n")


def emit_report(out: TextIO, report: CheckReport, args: argparse.Namespace) -> int:
    """Render any check report (JSON with ``--json``, else text) and return its exit code."""
    emit(out, report.to_dict(), as_json=args.json, text=report.render_text())
    return EXIT_OK if report.passed else EXIT_FAILED


def add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def add_experiments_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiments", help="path to EXPERIMENTS.yaml (default from config)")


def load_experiments(config: Config, path: str | None) -> Experiments:
    from qcal.registry.experiments import load_experiments as _load

    return _load(config, Path(path) if path else None)


def seed_list(text: str) -> list[int]:
    """argparse type for ``--seeds 0,1,2``: non-empty, integer, no duplicates."""
    parts = [p.strip() for p in text.split(",")]
    try:
        seeds = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"seeds must be comma-separated integers, got {text!r}"
        ) from None
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"duplicate seeds in {text!r}")
    return seeds
