"""The EvalLoop contract, as checks any loop can be run through (Ian's included).

``python -m qcal_lab check-eval-loop`` runs them on the configured loop; the contract suite
(``tests/contract``) runs them on the fixture stand-in too, which validates the checks
themselves. The inputs are a generated fixture and the fixture detector's predictions, so no
real metric value is ever computed or printed: only the shape and the stability of what the
loop returns are checked.

MUST problems mean the pipeline cannot rely on the loop; SHOULD findings are reported for
Ian to judge. ``CONTRACT_VERSION`` names the interface checked: 1 is today's ``EvalLoop``
(``qcal_lab.evaluation``); version 2 (docs/changes/evalloop-v2.md) gets its own checks once
Ian accepts it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from qcal.config import Config
from qcal.log import get_logger
from qcal.protocols import ImageDetections
from qcal.reports import FAIL, PASS
from qcal_lab.calib.two_threshold import keep_class
from qcal_lab.config import LabConfig
from qcal_lab.data.coco import GroundTruth, parse_coco
from qcal_lab.data.fixture import build_fixture, fixture_bytes
from qcal_lab.evaluation import (
    STAGES,
    EvalLoop,
    HandwrittenMissingError,
    check_metrics,
    check_targets,
    load_eval_loop,
)
from qcal_lab.models import build_detector

_log = get_logger("lab.eval_contract")

CONTRACT_VERSION: Final = 1
#: A threshold above every score: the objective must cope with a class that kept nothing.
EMPTY_THRESHOLD: Final = 1.5
NOT_ASSESSED: Final = "NOT ASSESSED"


def contract_fixture(settings: Mapping[str, Any]) -> GroundTruth:
    """The generated fixture the checks run on, from the smoke settings (``tooling.toml``)."""
    document = build_fixture(
        images=int(settings["images"]),
        categories=int(settings["categories"]),
        max_objects_per_image=int(settings["max_objects_per_image"]),
        seed=int(settings["seed"]),
        width=int(settings["image_width"]),
        height=int(settings["image_height"]),
    )
    return parse_coco(document, sha256=hashlib.sha256(fixture_bytes(document)).hexdigest())


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


class _Raised(Exception):
    """A loop or factory raised during a probe; carried as a MUST finding."""


def _probe(what: str, call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception as exc:
        raise _Raised(f"{what} raised {type(exc).__name__}: {exc}") from exc


def must_problems(
    build: Callable[[], object],  # a factory may return anything; checking that is the point
    predictions: Sequence[ImageDetections],
    ground_truth: GroundTruth,
) -> list[str]:
    """Violations of the MUST part of the contract; empty when the loop honours it.

    Every call into the loop or its factory is probed: one that raises, on any call, is a
    MUST finding, never an exception out of the checks.
    """
    problems: list[str] = []
    try:
        _must(build, predictions, ground_truth, problems)
    except _Raised as raised:
        problems.append(str(raised))
    return problems


def _must(
    build: Callable[[], object],
    predictions: Sequence[ImageDetections],
    ground_truth: GroundTruth,
    problems: list[str],
) -> None:
    loop = _probe("the factory", build)
    if not isinstance(loop, EvalLoop):
        problems.append("the factory did not return an EvalLoop")
        return
    before = copy.deepcopy((list(predictions), ground_truth))
    first = _probe("the loop", lambda: outputs(loop, predictions, ground_truth))
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
    again = _probe("the loop's second call", lambda: outputs(loop, predictions, ground_truth))
    if canonical(again) != canonical(first):
        problems.append("two calls on the same loop disagree")
    second = _probe("the factory's second call", build)
    if not isinstance(second, EvalLoop):
        problems.append("the factory's second call did not return an EvalLoop")
        return
    other = _probe("the second loop", lambda: outputs(second, predictions, ground_truth))
    if canonical(other) != canonical(first):
        problems.append("two loops from the same factory disagree")


def should_findings(
    loop: EvalLoop, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
) -> list[str]:
    """What the contract allows but Ian should know of: reported, never a failure."""
    findings: list[str] = []
    result = outputs(loop, predictions, ground_truth)
    for key, value in result["objectives"].items():
        if key.endswith(":all") and isinstance(value, float) and not math.isfinite(value):
            findings.append(f"objective {key} is {value!r} on a class with detections")
    return findings


@dataclass
class ContractReport:
    loop: str
    verdict: str
    version: int = CONTRACT_VERSION
    must: list[str] = field(default_factory=list)
    should: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop": self.loop,
            "contract_version": self.version,
            "verdict": self.verdict,
            "must": self.must,
            "should": self.should,
        }

    def render_text(self) -> str:
        lines = [f"evaluation loop contract v{self.version} ({self.loop}): {self.verdict}"]
        lines += [f"  MUST   {p}" for p in self.must]
        lines += [f"  SHOULD {f}" for f in self.should]
        return "\n".join(lines)


def check_eval_loop(
    lab: LabConfig,
    qcal_config: Config,
    settings: Mapping[str, Any],
    *,
    fixture: bool = False,
) -> ContractReport:
    """Run the contract on the configured loop (Ian's), or on the fixture stand-in.

    NOT ASSESSED while Ian's loop does not exist: agents never write it (CLAUDE.md rule 4).
    """
    ground_truth = contract_fixture(settings)
    predictions = fixture_inputs(lab, ground_truth)
    if fixture:
        from qcal_lab.fixture_eval import build as build_fixture_loop

        name = "qcal_lab.fixture_eval"

        def build() -> EvalLoop:
            return build_fixture_loop({})
    else:
        name = lab.text("eval_loop.module")
        try:
            load_eval_loop(lab, qcal_config)
        except HandwrittenMissingError:
            _log.info("%s does not exist yet: the contract is not assessed", name)
            return ContractReport(name, NOT_ASSESSED)

        def build() -> EvalLoop:
            return load_eval_loop(lab, qcal_config).loop

    must = must_problems(build, predictions, ground_truth)
    should: list[str] = []
    if not must:
        try:
            should = should_findings(build(), predictions, ground_truth)
        except Exception as exc:  # noqa: BLE001 - a loop raising here breaks the MUST part
            must = [f"the loop raised {type(exc).__name__}: {exc} (while its SHOULD part ran)"]
    report = ContractReport(name, FAIL if must else PASS, must=must, should=should)
    _log.info("evaluation loop contract: %s", report.verdict)
    return report


__all__ = [
    "CONTRACT_VERSION",
    "EMPTY_THRESHOLD",
    "FAIL",
    "NOT_ASSESSED",
    "PASS",
    "ContractReport",
    "canonical",
    "check_eval_loop",
    "contract_fixture",
    "fixture_inputs",
    "must_problems",
    "outputs",
    "should_findings",
]
