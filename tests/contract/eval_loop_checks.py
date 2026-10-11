"""The EvalLoop contract, as checks any loop can be run through (Ian's included).

MUST checks are asserted by tests/contract/test_eval_loop_contract.py: the pipeline relies on
them. The inputs are the synthetic fixture and the fixture detector's predictions, so no
real metric value is ever computed or printed here; only the shape and the stability of what
the loop returns are checked. PR-F moves these checks into ``qcal_lab.evaluation`` for a
``python -m qcal_lab check-eval-loop`` command; until then they live with their test.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

from qcal.protocols import ImageDetections
from qcal_lab.calib.two_threshold import keep_class
from qcal_lab.config import LabConfig
from qcal_lab.data.coco import GroundTruth
from qcal_lab.evaluation import STAGES, EvalLoop, check_metrics, check_targets
from qcal_lab.models import build_detector

#: A threshold above every score: the objective must cope with a class that kept nothing.
EMPTY_THRESHOLD = 1.5


def fixture_inputs(lab: LabConfig, ground_truth: GroundTruth) -> tuple[ImageDetections, ...]:
    detector = build_detector(lab, "fixture", ground_truth, precision="fp32")
    return tuple(detector.predict(sorted(ground_truth.images)))


def outputs(
    loop: EvalLoop, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
) -> dict[str, Any]:
    """Everything the loop returns for the fixture, in a canonical, comparable form."""
    objectives: dict[str, Any] = {}
    for label in range(ground_truth.num_classes):
        for stage in STAGES:
            for name, threshold in (("all", 0.0), ("empty", EMPTY_THRESHOLD)):
                subset = keep_class(predictions, label, threshold)
                value = loop.threshold_objective(subset, ground_truth, label=label, stage=stage)
                objectives[f"{label}:{stage}:{name}"] = value
    return {
        "targets": {k: list(v) for k, v in loop.targets(predictions, ground_truth).items()},
        "objectives": objectives,
        "metrics": dict(loop.metrics(predictions, ground_truth)),
    }


def canonical(data: Any) -> str:
    def encode(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)  # NaN is allowed for an objective; JSON has no NaN
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [encode(v) for v in value]
        return value

    return json.dumps(encode(data), sort_keys=True, separators=(",", ":"))


def must_problems(
    build: Callable[[], EvalLoop],
    predictions: Sequence[ImageDetections],
    ground_truth: GroundTruth,
) -> list[str]:
    """Violations of the MUST part of the contract; empty when the loop honours it."""
    problems: list[str] = []
    loop = build()
    if not isinstance(loop, EvalLoop):
        return ["the factory did not return an EvalLoop"]
    before = copy.deepcopy((list(predictions), ground_truth))
    try:
        first = outputs(loop, predictions, ground_truth)
    except Exception as exc:  # noqa: BLE001 - a raising loop is a finding, not a test crash
        return [f"the loop raised {type(exc).__name__}: {exc}"]
    if (list(predictions), ground_truth) != before:
        problems.append("the loop mutated its inputs")
    try:
        check_targets(predictions, first["targets"])
    except ValueError as exc:
        problems.append(f"targets: {exc}")
    for key, value in first["objectives"].items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            problems.append(f"objective {key} returned {value!r}, not a number")
    try:
        check_metrics(first["metrics"])
    except ValueError as exc:
        problems.append(f"metrics: {exc}")
    if canonical(outputs(loop, predictions, ground_truth)) != canonical(first):
        problems.append("two calls on the same loop disagree")
    if canonical(outputs(build(), predictions, ground_truth)) != canonical(first):
        problems.append("two loops from the same factory disagree")
    return problems
