"""A deterministic synthetic COCO-format dataset for the smoke test and the unit tests.

It has no pixels. The fixture detector and the fixture evaluation loop accept only this
dataset, and real detectors need image files, so nothing computed on it can pass as a COCO
result. It is generated on demand and never committed.
"""

from __future__ import annotations

import random
from typing import Any, Final

from qcal_lab.data.coco import GroundTruth

FIXTURE_DESCRIPTION: Final = "qcal_lab synthetic fixture (not a dataset; no pixels)"
_MIN_BOX_FRACTION: Final = 0.1
_MAX_BOX_FRACTION: Final = 0.5


def build_fixture(
    *,
    images: int,
    categories: int,
    max_objects_per_image: int,
    seed: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """A COCO-format document with ``images`` images and 1..max objects each."""
    if min(images, categories, max_objects_per_image, width, height) <= 0:
        raise ValueError("fixture sizes must be positive")
    rng = random.Random(f"qcal_lab-fixture:{seed}")  # noqa: S311 - synthetic data, not security
    doc_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for index in range(images):
        image_id = index + 1
        doc_images.append(
            {
                "id": image_id,
                "file_name": f"fixture_{image_id:04d}.jpg",
                "width": width,
                "height": height,
            }
        )
        for _ in range(rng.randint(1, max_objects_per_image)):
            w = rng.uniform(_MIN_BOX_FRACTION, _MAX_BOX_FRACTION) * width
            h = rng.uniform(_MIN_BOX_FRACTION, _MAX_BOX_FRACTION) * height
            x, y = rng.uniform(0, width - w), rng.uniform(0, height - h)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": rng.randint(1, categories),
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
    return {
        "info": {"description": FIXTURE_DESCRIPTION},
        "images": doc_images,
        "annotations": annotations,
        "categories": [{"id": c, "name": f"class_{c}"} for c in range(1, categories + 1)],
    }


def is_fixture(ground_truth: GroundTruth) -> bool:
    return ground_truth.description == FIXTURE_DESCRIPTION


class NotFixtureError(ValueError):
    """A fixture-only component was given a real dataset."""


def require_fixture(ground_truth: GroundTruth, component: str) -> None:
    if not is_fixture(ground_truth):
        raise NotFixtureError(
            f"{component} accepts only the synthetic fixture; it never runs on a real dataset"
        )


__all__ = [
    "FIXTURE_DESCRIPTION",
    "NotFixtureError",
    "build_fixture",
    "is_fixture",
    "require_fixture",
]
