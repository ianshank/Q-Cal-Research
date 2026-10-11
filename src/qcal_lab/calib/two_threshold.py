"""Class-wise calibration with two thresholds (Kuzucu et al., arXiv:2405.20459, Alg. A.1/A.2).

Training, Alg. A.1:

1. Select a calibration threshold u_c per class (LRP with tau = 0 in the paper).
2. Keep the detections with score >= u_c.
3. Fit a calibrator per class on those detections; ``scope = "global"`` fits one for all
   classes. A class with no detections left gets the identity (App. C.3).
4. Calibrate the selection split.
5. Select an operating threshold v_c per class on the calibrated scores.

Inference, Alg. A.2: drop scores below u_c, calibrate, then drop calibrated scores below v_c.

The calibrator fits on the fit role; both thresholds are selected on the select role (u_c on
raw scores, v_c on calibrated scores). The pipeline refuses to pass the evaluate split here.
The objective and the matching targets come from Ian's evaluation loop
(:mod:`qcal_lab.evaluation`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from qcal.log import get_logger
from qcal.protocols import Detection, ImageDetections
from qcal_lab.calib.base import (
    CalibrationError,
    IdentityCalibrator,
    SerializableCalibrator,
    load_calibrator,
)
from qcal_lab.calib.thresholds import select_threshold
from qcal_lab.data.coco import GroundTruth
from qcal_lab.evaluation import EvalLoop, check_targets

_log = get_logger("lab.calib.two_threshold")

SCOPES: Final = ("per_class", "global")
GLOBAL_KEY: Final = -1


@dataclass(frozen=True)
class SplitData:
    split: str
    predictions: tuple[ImageDetections, ...]
    ground_truth: GroundTruth


@dataclass(frozen=True)
class ThresholdedCalibration:
    """Fitted calibrators with their calibration (u_c) and operating (v_c) thresholds."""

    scope: str
    calibration_thresholds: Mapping[int, float]
    operating_thresholds: Mapping[int, float]
    calibrators: Mapping[int, SerializableCalibrator]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise CalibrationError(f"scope must be one of {SCOPES}, got {self.scope!r}")
        if self.scope == "global" and set(self.calibrators) != {GLOBAL_KEY}:
            raise CalibrationError("a global calibration has exactly one calibrator")

    def calibrator_for(self, label: int) -> SerializableCalibrator:
        key = GLOBAL_KEY if self.scope == "global" else label
        try:
            return self.calibrators[key]
        except KeyError:
            raise CalibrationError(f"no calibrator for class {label}") from None

    def _threshold(self, table: Mapping[int, float], label: int, which: str) -> float:
        try:
            return table[label]
        except KeyError:
            raise CalibrationError(f"no {which} threshold for class {label}") from None

    def calibrate(self, images: Sequence[ImageDetections]) -> tuple[ImageDetections, ...]:
        """Alg. A.2 lines 2 and 3: drop scores below u_c, then calibrate."""
        out: list[ImageDetections] = []
        for image in images:
            kept: list[Detection] = []
            for det in image.detections:
                u = self._threshold(self.calibration_thresholds, det.label, "calibration")
                if det.score >= u:
                    score = self.calibrator_for(det.label).transform([det.score])[0]
                    kept.append(Detection(det.box_xyxy, score, det.label, det.logit))
            out.append(ImageDetections(image.image_id, tuple(kept)))
        return tuple(out)

    def apply(self, images: Sequence[ImageDetections]) -> tuple[ImageDetections, ...]:
        """Alg. A.2: calibrate, then drop calibrated scores below v_c."""
        return apply_thresholds(self.calibrate(images), self.operating_thresholds)

    def to_dict(self) -> dict[str, Any]:
        def table(values: Mapping[int, float]) -> dict[str, float]:
            return {str(k): v for k, v in sorted(values.items())}

        return {
            "scope": self.scope,
            "calibration_thresholds": table(self.calibration_thresholds),
            "operating_thresholds": table(self.operating_thresholds),
            "calibrators": {str(k): c.to_dict() for k, c in sorted(self.calibrators.items())},
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ThresholdedCalibration:
        def table(key: str) -> dict[int, float]:
            raw = data.get(key)
            if not isinstance(raw, Mapping):
                raise CalibrationError(f"saved calibration needs a {key!r} table")
            return {int(k): float(v) for k, v in raw.items()}

        saved = data.get("calibrators")
        if not isinstance(saved, Mapping):
            raise CalibrationError("saved calibration needs a 'calibrators' table")
        provenance = data.get("provenance", {})
        return cls(
            scope=str(data.get("scope")),
            calibration_thresholds=table("calibration_thresholds"),
            operating_thresholds=table("operating_thresholds"),
            calibrators={int(k): load_calibrator(v) for k, v in saved.items()},
            provenance=dict(provenance) if isinstance(provenance, Mapping) else {},
        )


def keep_class(
    images: Sequence[ImageDetections], label: int, threshold: float
) -> tuple[ImageDetections, ...]:
    """Every image (so missed objects still count), with only ``label`` at ``threshold``."""
    return tuple(
        ImageDetections(
            image.image_id,
            tuple(d for d in image.detections if d.label == label and d.score >= threshold),
        )
        for image in images
    )


def apply_thresholds(
    images: Sequence[ImageDetections], thresholds: Mapping[int, float]
) -> tuple[ImageDetections, ...]:
    out: list[ImageDetections] = []
    for image in images:
        kept: list[Detection] = []
        for det in image.detections:
            if det.label not in thresholds:
                raise CalibrationError(f"no threshold for class {det.label}")
            if det.score >= thresholds[det.label]:
                kept.append(det)
        out.append(ImageDetections(image.image_id, tuple(kept)))
    return tuple(out)


def select_class_thresholds(
    data: SplitData,
    labels: Sequence[int],
    candidates: Sequence[float],
    eval_loop: EvalLoop,
    stage: str,
) -> tuple[dict[int, float], list[int]]:
    """Per class, the candidate minimising the loop's objective; plus classes that fell back."""
    chosen: dict[int, float] = {}
    fallbacks: list[int] = []
    for label in labels:

        def objective(threshold: float, label: int = label) -> float:
            subset = keep_class(data.predictions, label, threshold)
            return eval_loop.threshold_objective(
                subset, data.ground_truth, label=label, stage=stage
            )

        chosen[label], fell_back = select_threshold(candidates, objective)
        if fell_back:
            fallbacks.append(label)
    if fallbacks:
        _log.warning("%s thresholds fell back for %d class(es)", stage, len(fallbacks))
    return chosen, fallbacks


def _samples_by_key(
    images: Sequence[ImageDetections], targets: Mapping[str, Sequence[float]], scope: str
) -> dict[int, tuple[list[float], list[float]]]:
    samples: dict[int, tuple[list[float], list[float]]] = {}
    for image in images:
        for det, target in zip(image.detections, targets[image.image_id], strict=True):
            key = GLOBAL_KEY if scope == "global" else det.label
            scores, values = samples.setdefault(key, ([], []))
            scores.append(det.score)
            values.append(target)
    return samples


def train_calibration(
    *,
    fit: SplitData,
    calibration_threshold_data: SplitData,
    operating_threshold_data: SplitData,
    build: Callable[[], SerializableCalibrator],
    scope: str,
    candidates: Sequence[float],
    eval_loop: EvalLoop,
) -> ThresholdedCalibration:
    """Alg. A.1 with the calibrator from ``build`` and the objective from ``eval_loop``."""
    if scope not in SCOPES:
        raise CalibrationError(f"scope must be one of {SCOPES}, got {scope!r}")
    labels = list(range(fit.ground_truth.num_classes))
    u, u_fallbacks = select_class_thresholds(
        calibration_threshold_data, labels, candidates, eval_loop, "calibration"
    )
    thresholded = apply_thresholds(fit.predictions, u)
    targets = check_targets(thresholded, eval_loop.targets(thresholded, fit.ground_truth))
    samples = _samples_by_key(thresholded, targets, scope)
    keys = [GLOBAL_KEY] if scope == "global" else labels
    calibrators: dict[int, SerializableCalibrator] = {}
    identity: list[int] = []
    for key in keys:
        scores, values = samples.get(key, ([], []))
        if not scores:
            calibrators[key] = IdentityCalibrator()  # App. C.3: nothing left to fit on
            identity.append(key)
            continue
        calibrator = build()
        calibrator.fit(scores, values)
        calibrators[key] = calibrator
    provisional = ThresholdedCalibration(scope, u, dict.fromkeys(labels, 0.0), calibrators)
    calibrated = SplitData(
        operating_threshold_data.split,
        provisional.calibrate(operating_threshold_data.predictions),
        operating_threshold_data.ground_truth,
    )
    v, v_fallbacks = select_class_thresholds(calibrated, labels, candidates, eval_loop, "operating")
    provenance = {
        "procedure": "arXiv:2405.20459 Alg. A.1",
        "fit_split": fit.split,
        "calibration_threshold_split": calibration_threshold_data.split,
        "operating_threshold_split": operating_threshold_data.split,
        "fit_detections": {str(k): len(samples.get(k, ([], []))[0]) for k in keys},
        "identity_calibrators": identity,
        "calibration_threshold_fallbacks": u_fallbacks,
        "operating_threshold_fallbacks": v_fallbacks,
        "candidates": list(candidates),
    }
    return ThresholdedCalibration(scope, u, v, calibrators, provenance)


__all__ = [
    "GLOBAL_KEY",
    "SCOPES",
    "SplitData",
    "ThresholdedCalibration",
    "apply_thresholds",
    "keep_class",
    "select_class_thresholds",
    "train_calibration",
]
