"""A deterministic stand-in detector for the synthetic fixture (smoke test, unit tests).

It perturbs each ground-truth box and gives it a score that grows with the box's quality,
inflated by ``overconfidence`` so that calibrators have something to correct. It also adds
false positives and occasional wrong labels. It refuses every dataset except the fixture.
Randomness is seeded per (seed, image id), so predictions do not depend on the order or
batching of ``predict`` calls.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from qcal.protocols import Detection, ImageDetections
from qcal_lab.data.coco import GroundTruth, ImageInfo
from qcal_lab.data.fixture import require_fixture
from qcal_lab.models.base import (
    DETECTORS,
    DetectorContext,
    DetectorError,
    setting_int,
    setting_number,
)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class FixtureDetector:
    def __init__(
        self,
        name: str,
        ground_truth: GroundTruth,
        *,
        seed: int,
        detect_probability: float,
        box_jitter: float,
        label_flip_probability: float,
        false_positives_per_image: int,
        overconfidence: float,
    ) -> None:
        require_fixture(ground_truth, "the fixture detector")
        self.name = name
        self.ground_truth = ground_truth
        self.seed = seed
        self.detect_probability = detect_probability
        self.box_jitter = box_jitter
        self.label_flip_probability = label_flip_probability
        self.false_positives_per_image = false_positives_per_image
        self.overconfidence = overconfidence

    def _box(
        self, rng: random.Random, box: tuple[float, float, float, float], info: ImageInfo
    ) -> tuple[tuple[float, float, float, float], float]:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        shifts = [rng.uniform(-self.box_jitter, self.box_jitter) for _ in range(4)]
        nx1 = _clip(x1 + shifts[0] * w, 0.0, float(info.width))
        ny1 = _clip(y1 + shifts[1] * h, 0.0, float(info.height))
        nx2 = _clip(x2 + shifts[2] * w, nx1, float(info.width))
        ny2 = _clip(y2 + shifts[3] * h, ny1, float(info.height))
        spread = sum(abs(s) for s in shifts) / (4 * self.box_jitter) if self.box_jitter else 0.0
        quality = _clip(1.0 - spread, 0.0, 1.0)
        return (nx1, ny1, nx2, ny2), quality

    def _image(self, image_id: str) -> ImageDetections:
        info = self.ground_truth.images.get(image_id)
        if info is None:
            raise DetectorError(f"image {image_id} is not in the fixture")
        rng = random.Random(f"{self.seed}:{image_id}")  # noqa: S311 - synthetic data
        classes = self.ground_truth.num_classes
        detections: list[Detection] = []
        for gt in self.ground_truth.boxes.get(image_id, ()):
            if rng.random() > self.detect_probability:
                continue
            box, quality = self._box(rng, gt.box_xyxy, info)
            label = gt.label
            if rng.random() < self.label_flip_probability:
                label = rng.randrange(classes)
            score = _clip(quality**self.overconfidence * rng.uniform(0.85, 1.0), 0.0, 1.0)
            detections.append(Detection(box, score, label))
        for _ in range(self.false_positives_per_image):
            w = rng.uniform(0.05, 0.3) * info.width
            h = rng.uniform(0.05, 0.3) * info.height
            x, y = rng.uniform(0, info.width - w), rng.uniform(0, info.height - h)
            score = _clip(rng.uniform(0.0, 0.6) ** self.overconfidence, 0.0, 1.0)
            detections.append(Detection((x, y, x + w, y + h), score, rng.randrange(classes)))
        detections.sort(key=lambda d: -d.score)
        return ImageDetections(image_id, tuple(detections))

    def predict(self, image_ids: Sequence[str]) -> list[ImageDetections]:
        return [self._image(i) for i in image_ids]


def _build(context: DetectorContext) -> FixtureDetector:
    s = context.settings
    return FixtureDetector(
        context.name,
        context.ground_truth,
        seed=setting_int(s, "seed"),
        detect_probability=setting_number(s, "detect_probability", low=0.0, high=1.0),
        box_jitter=setting_number(s, "box_jitter", low=0.0, high=1.0),
        label_flip_probability=setting_number(s, "label_flip_probability", low=0.0, high=1.0),
        false_positives_per_image=setting_int(s, "false_positives_per_image"),
        overconfidence=setting_number(s, "overconfidence", low=1e-3, high=10.0),
    )


DETECTORS.register("fixture", _build)

__all__ = ["FixtureDetector"]
