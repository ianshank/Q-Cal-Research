"""A stand-in evaluation loop for the synthetic fixture only (smoke test and unit tests).

This is NOT the evaluation protocol and computes none of the paper's metrics. It refuses
every dataset except the synthetic fixture, and every metric name it reports starts with
``smoke_``. Reported metrics come only from Ian's hand-written loop (CLAUDE.md rule 4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from qcal.protocols import ImageDetections
from qcal_lab.data.coco import Box, GroundTruth
from qcal_lab.data.fixture import require_fixture

_HIT_IOU: Final = 0.5


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class FixtureEvalLoop:
    """Targets are the best IoU with a same-class object; nothing is matched one-to-one."""

    name = "fixture"

    def _best_iou(self, gt: GroundTruth, image_id: str, box: Box, label: int) -> float:
        boxes = [g.box_xyxy for g in gt.boxes.get(image_id, ()) if g.label == label]
        return max((box_iou(box, g) for g in boxes), default=0.0)

    def targets(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> dict[str, list[float]]:
        require_fixture(ground_truth, "the fixture evaluation loop")
        return {
            image.image_id: [
                self._best_iou(ground_truth, image.image_id, d.box_xyxy, d.label)
                for d in image.detections
            ]
            for image in predictions
        }

    def threshold_objective(
        self,
        predictions: Sequence[ImageDetections],
        ground_truth: GroundTruth,
        *,
        label: int,
        stage: str,  # noqa: ARG002 - one stand-in objective for both stages
    ) -> float:
        """Share of poor detections and missed objects of one class (NaN when it has neither)."""
        targets = self.targets(predictions, ground_truth)
        poor = sum(t < _HIT_IOU for values in targets.values() for t in values)
        detections = sum(len(v) for v in targets.values())
        objects = 0
        missed = 0
        for image in predictions:
            for gt in ground_truth.boxes.get(image.image_id, ()):
                if gt.label != label:
                    continue
                objects += 1
                if all(box_iou(d.box_xyxy, gt.box_xyxy) < _HIT_IOU for d in image.detections):
                    missed += 1
        total = detections + objects
        return (poor + missed) / total if total else float("nan")

    def metrics(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> dict[str, float]:
        targets = self.targets(predictions, ground_truth)
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


def build(options: Mapping[str, Any]) -> FixtureEvalLoop:  # noqa: ARG001 - no options
    return FixtureEvalLoop()


__all__ = ["FixtureEvalLoop", "box_iou", "build"]
