"""A stand-in evaluation loop for the synthetic fixture only (smoke test and unit tests).

This is NOT the evaluation protocol and computes none of the paper's metrics. The loop refuses
every dataset except an unmodified generated fixture (recognised by content digest), and every
metric name it reports starts with ``smoke_``. Reported metrics come only from Ian's
hand-written loop (CLAUDE.md rule 4); :func:`qcal_lab.evaluation.load_eval_loop` accepts this
module and Ian-only modules, nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qcal.config import ConfigError
from qcal.protocols import ImageDetections
from qcal_lab.config import load_tooling
from qcal_lab.data.coco import Box, GroundTruth
from qcal_lab.data.fixture import require_fixture


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_iou_targets(
    predictions: Sequence[ImageDetections], ground_truth: GroundTruth
) -> dict[str, list[float]]:
    """Each detection's best IoU with a same-class object; nothing is matched one-to-one."""
    out: dict[str, list[float]] = {}
    for image in predictions:
        objects = ground_truth.boxes.get(image.image_id, ())
        out[image.image_id] = [
            max(
                (box_iou(d.box_xyxy, g.box_xyxy) for g in objects if g.label == d.label),
                default=0.0,
            )
            for d in image.detections
        ]
    return out


def tooling_hit_iou() -> float:
    """The fixture loop's notion of "found" (``tooling.toml``); not an evaluation threshold."""
    value = float(load_tooling()["fixture_eval"]["hit_iou"])
    if not 0.0 < value <= 1.0:
        raise ConfigError(f"fixture_eval.hit_iou must be in (0, 1], got {value}")
    return value


def poor_or_missed_share(
    predictions: Sequence[ImageDetections],
    ground_truth: GroundTruth,
    label: int,
    *,
    hit_iou: float | None = None,
) -> float:
    """Share of poor detections and missed objects of one class (NaN when it has neither).

    A detection is poor, and an object missed, below ``hit_iou`` (default: the tooling value).
    """
    targets = best_iou_targets(predictions, ground_truth)
    threshold = tooling_hit_iou() if hit_iou is None else hit_iou
    poor = sum(t < threshold for values in targets.values() for t in values)
    detections = sum(len(v) for v in targets.values())
    objects = missed = 0
    for image in predictions:
        for gt in ground_truth.boxes.get(image.image_id, ()):
            if gt.label != label:
                continue
            objects += 1
            if all(box_iou(d.box_xyxy, gt.box_xyxy) < threshold for d in image.detections):
                missed += 1
    total = detections + objects
    return (poor + missed) / total if total else float("nan")


def smoke_metrics(
    predictions: Sequence[ImageDetections], ground_truth: GroundTruth
) -> dict[str, float]:
    targets = best_iou_targets(predictions, ground_truth)
    gaps = [
        abs(d.score - t)
        for image in predictions
        for d, t in zip(image.detections, targets[image.image_id], strict=True)
    ]
    return {
        "smoke_images": float(len(predictions)),
        "smoke_detections": float(len(gaps)),
        "smoke_mean_abs_gap": sum(gaps) / len(gaps) if gaps else 0.0,
    }


class FixtureEvalLoop:
    """The functions above, behind the EvalLoop interface, for the fixture only."""

    name = "fixture"

    def __init__(self, hit_iou: float | None = None) -> None:
        # Read once here: the loop itself must read no files while it evaluates.
        self.hit_iou = tooling_hit_iou() if hit_iou is None else hit_iou

    def targets(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> dict[str, list[float]]:
        require_fixture(ground_truth, "the fixture evaluation loop")
        return best_iou_targets(predictions, ground_truth)

    def threshold_objective(
        self,
        predictions: Sequence[ImageDetections],
        ground_truth: GroundTruth,
        *,
        label: int,
        stage: str,  # noqa: ARG002 - one stand-in objective for both stages
    ) -> float:
        require_fixture(ground_truth, "the fixture evaluation loop")
        return poor_or_missed_share(predictions, ground_truth, label, hit_iou=self.hit_iou)

    def metrics(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> dict[str, float]:
        require_fixture(ground_truth, "the fixture evaluation loop")
        return smoke_metrics(predictions, ground_truth)


def build(options: Mapping[str, Any]) -> FixtureEvalLoop:  # noqa: ARG001 - no options
    return FixtureEvalLoop()


__all__ = [
    "FixtureEvalLoop",
    "best_iou_targets",
    "box_iou",
    "build",
    "poor_or_missed_share",
    "smoke_metrics",
    "tooling_hit_iou",
]
