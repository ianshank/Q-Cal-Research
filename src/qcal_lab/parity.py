"""Oracle parity cases: outputs of ``fiveai/detection_calibration`` on identical inputs.

The oracle is licensed CC BY-NC-SA 4.0. It runs unmodified in its own environment (the P40,
CUDA 11.8; plan §5 Phase 1), and only its outputs are committed, under
``parity.fixtures_dir``. Nothing in this package imports it. ``qcal licenses`` allows the
import only under ``tests/parity/``.

A case is one JSON file, in the ``qcal_lab.parity_case`` format (version 1):

    {"format": "qcal_lab.parity_case", "version": 1,
     "kind": "calibrator", "calibrator": "isotonic",
     "oracle": {"repository": "fiveai/detection_calibration", "commit": "<sha>",
                "command": "<what was run>",
                "environment": {"python": "3.8.18", "torch": "1.13.1", "cuda": "11.8"}},
     "fit": {"scores": [...], "targets": [...]},
     "transform": {"scores": [...], "expected": [...]},
     "tolerance": 1e-6}

    {"format": "qcal_lab.parity_case", "version": 1,
     "kind": "metric", "metric": "LaECE0", "oracle": {...},
     "predictions": "<predictions .jsonl, relative to the case>",
     "ground_truth": "<COCO json, relative to the case>",
     "expected": 12.34, "tolerance": 1e-6}

``oracle.environment`` records the oracle's own environment (its interpreter and library
versions), so a parity mismatch can be traced to a version difference.

Calibrator cases test this package. Metric cases test Ian's hand-written evaluation loop:
a mismatch is reported, never fixed by editing his file (CLAUDE.md rule 4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from qcal.config import Config
from qcal_lab.calib import build_calibrator
from qcal_lab.config import LabConfig
from qcal_lab.data.coco import load_coco
from qcal_lab.evaluation import EvalLoop, load_eval_loop
from qcal_lab.formats import open_envelope, read_json
from qcal_lab.predictions import read_predictions

CASE_FORMAT: Final = "qcal_lab.parity_case"
CASE_VERSION: Final = 1
KINDS: Final = ("calibrator", "metric")
_ORACLE_KEYS: Final = ("repository", "commit")
_ORACLE_ENVIRONMENT: Final = "environment"


class ParityError(ValueError):
    """A parity case file is malformed."""


@dataclass(frozen=True)
class OracleCase:
    path: Path
    kind: str
    data: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.path.stem

    def tolerance(self, default: float) -> float:
        """The case's tolerance: it may tighten ``parity.abs_tolerance``, never loosen it."""
        value = self.data.get("tolerance", default)
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise ParityError(f"{self.path.name}: tolerance must be a non-negative number")
        if value > default:
            raise ParityError(
                f"{self.path.name}: tolerance {value} is looser than parity.abs_tolerance {default}"
            )
        return float(value)


def _numbers(section: Mapping[str, Any], key: str, where: str) -> list[float]:
    values = section.get(key)
    if not isinstance(values, list) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in values
    ):
        raise ParityError(f"{where}: {key!r} must be a list of numbers")
    return [float(v) for v in values]


def load_case(path: Path) -> OracleCase:
    raw = read_json(path, error=ParityError)
    data = open_envelope(raw, CASE_FORMAT, CASE_VERSION, path.name, error=ParityError)
    if data.get("kind") not in KINDS:
        raise ParityError(f"{path.name}: 'kind' must be one of {KINDS}")
    oracle = data.get("oracle")
    if not isinstance(oracle, Mapping) or not all(
        isinstance(oracle.get(k), str) and oracle.get(k) for k in _ORACLE_KEYS
    ):
        raise ParityError(f"{path.name}: 'oracle' must record {', '.join(_ORACLE_KEYS)}")
    environment = oracle.get(_ORACLE_ENVIRONMENT)
    if not isinstance(environment, Mapping) or not environment:
        raise ParityError(
            f"{path.name}: 'oracle.{_ORACLE_ENVIRONMENT}' must record the oracle's interpreter "
            "and library versions"
        )
    return OracleCase(path, str(data["kind"]), data)


def load_cases(directory: Path) -> list[OracleCase]:
    if not directory.is_dir():
        return []
    return [load_case(p) for p in sorted(directory.glob("*.json"))]


def compare(actual: Sequence[float], expected: Sequence[float], tolerance: float) -> list[str]:
    """Human-readable mismatches beyond ``tolerance`` (absolute), at most ten."""
    if len(actual) != len(expected):
        return [f"{len(actual)} values, oracle has {len(expected)}"]
    problems = [
        f"[{i}] {a!r} vs oracle {e!r} (|diff| {abs(a - e):.3g})"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True))
        if not (math.isfinite(a) and abs(a - e) <= tolerance)
    ]
    return problems[:10]


def check_calibrator_case(case: OracleCase, lab: LabConfig) -> list[str]:
    where = case.path.name
    fit, transform = case.data.get("fit"), case.data.get("transform")
    name = case.data.get("calibrator")
    if not isinstance(fit, Mapping) or not isinstance(transform, Mapping):
        raise ParityError(f"{where}: needs 'fit' and 'transform' objects")
    if not isinstance(name, str):
        raise ParityError(f"{where}: needs a 'calibrator' name")
    calibrator = build_calibrator(name, lab)
    calibrator.fit(_numbers(fit, "scores", where), _numbers(fit, "targets", where))
    actual = list(calibrator.transform(_numbers(transform, "scores", where)))
    default = lab.config.float_value("parity.abs_tolerance")
    return compare(actual, _numbers(transform, "expected", where), case.tolerance(default))


def check_metric_case(case: OracleCase, eval_loop: EvalLoop, default_tolerance: float) -> list[str]:
    where = case.path.name
    metric = case.data.get("metric")
    expected = case.data.get("expected")
    if (
        not isinstance(metric, str)
        or isinstance(expected, bool)
        or not isinstance(expected, int | float)
    ):
        raise ParityError(f"{where}: needs a 'metric' name and a numeric 'expected'")
    paths = {}
    for key in ("predictions", "ground_truth"):
        value = case.data.get(key)
        if not isinstance(value, str):
            raise ParityError(f"{where}: needs a {key!r} path")
        paths[key] = case.path.parent / value
    _, predictions = read_predictions(paths["predictions"])
    ground_truth = load_coco(paths["ground_truth"])
    metrics = eval_loop.metrics(predictions, ground_truth)
    if metric not in metrics:
        return [f"the evaluation loop reports no {metric!r} (has: {sorted(metrics)})"]
    return compare([float(metrics[metric])], [float(expected)], case.tolerance(default_tolerance))


def check_metric_case_with_repository_loop(
    case: OracleCase, lab: LabConfig, qcal_config: Config
) -> list[str]:
    """Load the configured evaluation loop (Ian's) and check one metric case against it.

    Raises :class:`qcal_lab.evaluation.HandwrittenMissingError` while the loop is unwritten.
    """
    loop = load_eval_loop(lab, qcal_config).loop
    return check_metric_case(case, loop, lab.config.float_value("parity.abs_tolerance"))


__all__ = [
    "CASE_FORMAT",
    "CASE_VERSION",
    "KINDS",
    "OracleCase",
    "ParityError",
    "check_calibrator_case",
    "check_metric_case",
    "check_metric_case_with_repository_loop",
    "compare",
    "load_case",
    "load_cases",
]
