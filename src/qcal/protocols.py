"""Interfaces for the science components that Phase 1 implements.

Only the shapes live here, so infrastructure, tests and later implementations can
be wired by dependency injection. Implementations belong in the agent-owned
``qcal_lab`` package; everything under ``src/qcal/`` is the Ian-signed integrity layer.
No detection-calibration logic is implemented in Phase 0: that work is gated on G0
(plan §5) and parts of it are Ian-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    label: int
    logit: float | None = None  # raw score before sigmoid/softmax, for calibration metrics


@dataclass(frozen=True)
class ImageDetections:
    image_id: str
    detections: tuple[Detection, ...] = field(default_factory=tuple)


@runtime_checkable
class Detector(Protocol):
    """Produces raw detections; post-processing such as NMS is explicit and configurable."""

    name: str

    def predict(self, image_ids: Sequence[str]) -> Sequence[ImageDetections]: ...


@runtime_checkable
class Calibrator(Protocol):
    """Post-hoc confidence calibrator fitted on ``calibrator_fit_split`` only."""

    name: str

    def fit(self, scores: Sequence[float], targets: Sequence[float]) -> None: ...

    def transform(self, scores: Sequence[float]) -> Sequence[float]: ...


@runtime_checkable
class Metric(Protocol):
    """A named metric computed from predictions and ground truth."""

    name: str

    def compute(
        self, predictions: Sequence[ImageDetections], ground_truth: Mapping[str, Any]
    ) -> float: ...


@runtime_checkable
class Quantizer(Protocol):
    """Builds a quantized artifact (for example a TensorRT engine) and reports how it was built."""

    name: str

    def build(
        self, model_path: Path, calibration_images: Sequence[str], output: Path
    ) -> Mapping[str, Any]: ...
