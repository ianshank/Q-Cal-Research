"""Shared builders for the qcal_lab tests: tiny datasets, detections and fixture projects."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qcal.protocols import Detection, ImageDetections
from qcal_lab.config import LabConfig, load_lab_config, parse_lab_config
from qcal_lab.data.coco import GroundTruth, parse_coco
from qcal_lab.data.fixture import build_fixture, fixture_bytes
from qcal_lab.predictions import PredictionsHeader, PredictionSource
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


def ground_truth(**kwargs: Any) -> GroundTruth:
    return parse_coco(coco_doc(**kwargs), sha256="test")


FIXTURE_PARAMETERS: dict[str, int] = {
    "images": 6,
    "categories": 3,
    "max_objects_per_image": 3,
    "seed": 1,
    "width": 64,
    "height": 48,
}


def fixture_document(**overrides: int) -> dict[str, Any]:
    return build_fixture(**{**FIXTURE_PARAMETERS, **overrides})


def fixture_ground_truth(**overrides: int) -> GroundTruth:
    """A generated fixture, loaded the way a file would be (digest of its canonical bytes)."""
    document = fixture_document(**overrides)
    return parse_coco(document, sha256=hashlib.sha256(fixture_bytes(document)).hexdigest())


def det(score: float, label: int = 0, box: tuple[float, ...] = (0, 0, 10, 10)) -> Detection:
    x1, y1, x2, y2 = (float(v) for v in box)
    return Detection((x1, y1, x2, y2), score, label)


def image(image_id: str, *detections: Detection) -> ImageDetections:
    return ImageDetections(image_id, tuple(detections))


#: A predictions source for tests that do not care what produced the predictions.
SOURCE = PredictionSource(
    precision="fp32",
    target="torch_fp32",
    quant_path="none",
    shift="id",
    score_definition="test: hand-written detections",
)


def predictions_header(
    detector: str = "fixture",
    split: str = "val",
    stage: str = "raw",
    category_ids: Sequence[int] = (1, 3),
    *,
    dataset_sha256: str = "d" * 64,
    split_sha256: str = "s" * 64,
    source: PredictionSource = SOURCE,
) -> PredictionsHeader:
    return PredictionsHeader(
        detector, split, stage, dataset_sha256, split_sha256, tuple(category_ids), source
    )


def lab_config(root: Path, text: str = "") -> LabConfig:
    return parse_lab_config(root, text or None)


def fixture_project(workdir: Path) -> tuple[Path, list[str]]:
    """The smoke test's throwaway project (fixture dataset, manifests, cells), not yet run."""
    project = workdir / "fixture-project"
    cells = write_project(project, load_lab_config(workdir), sys.executable)
    return project, cells


class FakeTorch:
    """Enough of torch for the numerics regimes and the device record; torch is not a test
    dependency. Switches start at torch's defaults (TF32 on for cuDNN, off for matmul)."""

    def __init__(self, *, cuda_available: bool = True) -> None:
        self.__version__ = "2.9.0+cu128"
        self.version = SimpleNamespace(cuda="12.8")
        self.__config__ = SimpleNamespace(show=lambda: "PyTorch built with: CUDA 12.8")
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(
                allow_tf32=True, benchmark=False, deterministic=False, version=lambda: 91000
            ),
        )
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_properties=lambda index: SimpleNamespace(
                name=f"GPU {index}", uuid=f"GPU-{index:04d}", major=12, minor=0, total_memory=2**34
            ),
        )
        self.deterministic = False
        self.warn_only = False

    def use_deterministic_algorithms(self, mode: bool, *, warn_only: bool = False) -> None:
        self.deterministic, self.warn_only = mode, warn_only

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only
