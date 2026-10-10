"""A stand-in evaluation loop for the synthetic fixture only (smoke test and unit tests).

This is NOT the evaluation protocol and computes none of the paper's metrics. The loop refuses
every dataset except an unmodified generated fixture (recognised by content digest), and every
metric name it reports starts with ``smoke_``. Reported metrics come only from Ian's
hand-written loop (CLAUDE.md rule 4); :func:`qcal_lab.evaluation.load_eval_loop` accepts this
module and Ian-only modules, nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from qcal.protocols import ImageDetections
from qcal_lab.data.coco import Box, GroundTruth
from qcal_lab.data.fixture import require_fixture

_HIT_IOU: Final = 0.5  # a fixture-only notion of "found"; not an evaluation threshold


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


def poor_or_missed_share(
    predictions: Sequence[ImageDetections], ground_truth: GroundTruth, label: int
) -> float:
    """Share of poor detections and missed objects of one class (NaN when it has neither)."""
    targets = best_iou_targets(predictions, ground_truth)
    poor = sum(t < _HIT_IOU for values in targets.values() for t in values)
    detections = sum(len(v) for v in targets.values())
    objects = missed = 0
    for image in predictions:
        for gt in ground_truth.boxes.get(image.image_id, ()):
            if gt.label != label:
                continue
            objects += 1
            if all(box_iou(d.box_xyxy, gt.box_xyxy) < _HIT_IOU for d in image.detections):
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
        return poor_or_missed_share(predictions, ground_truth, label)

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
]
