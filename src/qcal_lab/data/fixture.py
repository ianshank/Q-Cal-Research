"""A deterministic synthetic COCO-format dataset for the smoke test and the unit tests.

It has no pixels. The fixture detector and the fixture evaluation loop accept only this
dataset, and real detectors need image files, so nothing computed on it can pass as a COCO
result. It is generated on demand and never committed.

A dataset is recognised as the fixture by content, not by a label: its ``info`` records the
generation parameters, and the file's sha256 must equal that of the document those
parameters regenerate (:func:`is_fixture`). Copying the description into real annotations,
or editing a generated fixture, does not pass.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
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
    parameters = {
        "images": images,
        "categories": categories,
        "max_objects_per_image": max_objects_per_image,
        "seed": seed,
        "width": width,
        "height": height,
    }
    return {
        "info": {"description": FIXTURE_DESCRIPTION, "fixture_parameters": parameters},
        "images": doc_images,
        "annotations": annotations,
        "categories": [{"id": c, "name": f"class_{c}"} for c in range(1, categories + 1)],
    }


def fixture_bytes(document: Mapping[str, Any]) -> bytes:
    """The canonical serialisation a fixture is written and recognised by."""
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_fixture(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture_bytes(document))


def is_fixture(ground_truth: GroundTruth) -> bool:
    """True only for an unmodified document that :func:`build_fixture` generated."""
    parameters = ground_truth.info.get("fixture_parameters")
    if ground_truth.description != FIXTURE_DESCRIPTION or not isinstance(parameters, Mapping):
        return False
    try:
        rebuilt = build_fixture(**{str(k): v for k, v in parameters.items()})
    except (TypeError, ValueError):
        return False
    return hashlib.sha256(fixture_bytes(rebuilt)).hexdigest() == ground_truth.sha256


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
    "fixture_bytes",
    "is_fixture",
    "require_fixture",
    "write_fixture",
]
