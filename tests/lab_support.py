"""Shared builders for the qcal_lab tests: tiny datasets, detections and fixture projects."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qcal.protocols import Detection, ImageDetections
from qcal_lab.config import LabConfig, load_lab_config, parse_lab_config
from qcal_lab.data.coco import GroundTruth, parse_coco
from qcal_lab.data.fixture import FIXTURE_DESCRIPTION
from qcal_lab.smoke import write_project


def coco_doc(
    images: int = 3,
    *,
    categories: Sequence[int] = (1, 3),
    description: str = "",
    boxes: Mapping[int, Sequence[tuple[int, list[float]]]] | None = None,
) -> dict[str, Any]:
    """A COCO document; ``boxes`` maps image id -> [(category id, [x, y, w, h])]."""
    boxes = boxes if boxes is not None else {1: [(1, [10.0, 10.0, 20.0, 20.0])]}
    annotations = [
        {"id": n, "image_id": image, "category_id": cat, "bbox": bbox, "iscrowd": 0}
        for n, (image, cat, bbox) in enumerate(
            ((i, c, b) for i, entries in boxes.items() for c, b in entries), start=1
        )
    ]
    return {
        "info": {"description": description},
        "images": [
            {"id": i, "file_name": f"{i:06d}.jpg", "width": 100, "height": 80}
            for i in range(1, images + 1)
        ],
        "annotations": annotations,
        "categories": [{"id": c, "name": f"c{c}"} for c in categories],
    }


def ground_truth(*, fixture: bool = False, **kwargs: Any) -> GroundTruth:
    description = FIXTURE_DESCRIPTION if fixture else ""
    return parse_coco(coco_doc(description=description, **kwargs), sha256="test")


def det(score: float, label: int = 0, box: tuple[float, ...] = (0, 0, 10, 10)) -> Detection:
    x1, y1, x2, y2 = (float(v) for v in box)
    return Detection((x1, y1, x2, y2), score, label)


def image(image_id: str, *detections: Detection) -> ImageDetections:
    return ImageDetections(image_id, tuple(detections))


def lab_config(root: Path, text: str = "") -> LabConfig:
    return parse_lab_config(root, text or None)


def fixture_project(workdir: Path) -> tuple[Path, list[str]]:
    """The smoke test's throwaway project (fixture dataset, manifests, cells), not yet run."""
    project = workdir / "fixture-project"
    cells = write_project(project, load_lab_config(workdir), sys.executable)
    return project, cells
