"""The two-threshold class-wise procedure (arXiv:2405.20459 Alg. A.1 and A.2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from qcal.protocols import ImageDetections
from qcal_lab.calib import build_calibrator
from qcal_lab.calib.base import CalibrationError, IdentityCalibrator
from qcal_lab.calib.two_threshold import (
    GLOBAL_KEY,
    SplitData,
    ThresholdedCalibration,
    apply_thresholds,
    keep_class,
    train_calibration,
)
from qcal_lab.data.coco import GroundTruth
from tests.lab_support import det, ground_truth, image, lab_config


class ScriptedLoop:
    """Targets equal the score minus 0.2; the objective prefers a chosen threshold per class."""

    def __init__(self, best: Mapping[tuple[str, int], float]) -> None:
        self.best = best
        self.calls: list[tuple[str, int, str]] = []

    def targets(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> dict[str, list[float]]:
        return {i.image_id: [max(d.score - 0.2, 0.0) for d in i.detections] for i in predictions}

    def threshold_objective(
        self,
        predictions: Sequence[ImageDetections],
        ground_truth: GroundTruth,
        *,
        label: int,
        stage: str,
    ) -> float:
        kept = [d for i in predictions for d in i.detections]
        assert all(d.label == label for d in kept)
        self.calls.append((stage, label, ",".join(sorted(ground_truth.images))))
        lowest = min((d.score for d in kept), default=1.0)
        target = self.best.get((stage, label))
        return float("nan") if target is None else abs(lowest - target)

    def metrics(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> dict[str, float]:
        return {"n": 0.0}


GT = ground_truth(images=3, categories=(1, 2, 3))
FIT = SplitData(
    "calibrator_fit_split",
    (
        image("1", det(0.9, 0), det(0.6, 0), det(0.3, 0), det(0.8, 1)),
        image("2", det(0.7, 0), det(0.5, 1), det(0.2, 1)),
    ),
    GT.subset(["1", "2"]),
)
SELECT = SplitData("val", (image("3", det(0.95, 0), det(0.4, 0), det(0.6, 1)),), GT.subset(["3"]))
GRID = (0.0, 0.25, 0.5, 0.75)


def _train(tmp_path: Path, scope: str, loop: ScriptedLoop, name: str = "isotonic"):
    lab = lab_config(tmp_path)
    return train_calibration(
        fit=FIT,
        calibration_threshold_data=FIT,
        operating_threshold_data=SELECT,
        build=lambda: build_calibrator(name, lab),
        scope=scope,
        candidates=GRID,
        eval_loop=loop,
    )


def test_alg_a1_per_class(tmp_path: Path) -> None:
    # Class 1 scores are 0.8, 0.5, 0.2: thresholds 0.25 and 0.5 both keep lowest 0.5 (a tie,
    # so the earlier candidate wins).
    loop = ScriptedLoop({("calibration", 0): 0.5, ("calibration", 1): 0.5, ("operating", 0): 0.3})
    cal = _train(tmp_path, "per_class", loop)
    assert cal.calibration_thresholds == {0: 0.5, 1: 0.25, 2: 0.0}
    assert cal.provenance["calibration_threshold_fallbacks"] == [2]  # nothing to choose from
    assert cal.provenance["fit_detections"] == {"0": 3, "1": 2, "2": 0}
    assert cal.provenance["identity_calibrators"] == [2]  # App. C.3
    assert isinstance(cal.calibrators[2], IdentityCalibrator)
    # Thresholds are chosen on the fit split, then the select split; never anything else.
    assert {(stage, ids) for stage, _, ids in loop.calls} == {
        ("calibration", "1,2"),
        ("operating", "3"),
    }
    assert set(cal.operating_thresholds) == {0, 1, 2}


def test_alg_a2_thresholds_calibrates_then_thresholds_again(tmp_path: Path) -> None:
    cal = ThresholdedCalibration(
        "per_class",
        calibration_thresholds={0: 0.5, 1: 0.0},
        operating_thresholds={0: 0.6, 1: 0.0},
        calibrators={0: _scaled(tmp_path, 0.9), 1: IdentityCalibrator()},
    )
    out = cal.apply([image("9", det(0.9, 0), det(0.6, 0), det(0.4, 0), det(0.1, 1))])
    # 0.4 < u_0 is dropped before calibration; 0.6 -> 0.54 < v_0 is dropped after it.
    assert [(d.label, round(d.score, 6)) for d in out[0].detections] == [(0, 0.81), (1, 0.1)]


def _scaled(tmp_path: Path, factor: float):
    """An isotonic calibrator equal to ``factor * score`` on [0, 1]."""
    model = build_calibrator("isotonic", lab_config(tmp_path))
    model.fit([0.0, 1.0], [0.0, factor])
    return model


def test_global_scope_fits_one_calibrator(tmp_path: Path) -> None:
    cal = _train(tmp_path, "global", ScriptedLoop({}), name="platt")
    assert set(cal.calibrators) == {GLOBAL_KEY}
    assert cal.calibrator_for(0) is cal.calibrator_for(2)
    assert cal.provenance["fit_detections"] == {str(GLOBAL_KEY): 7}


def test_global_scope_with_nothing_left_uses_identity(tmp_path: Path) -> None:
    empty = SplitData("calibrator_fit_split", (image("1"),), GT.subset(["1"]))
    cal = train_calibration(
        fit=empty,
        calibration_threshold_data=empty,
        operating_threshold_data=SELECT,
        build=lambda: build_calibrator("platt", lab_config(tmp_path)),
        scope="global",
        candidates=GRID,
        eval_loop=ScriptedLoop({}),
    )
    assert isinstance(cal.calibrators[GLOBAL_KEY], IdentityCalibrator)


def test_saved_calibration_round_trips(tmp_path: Path) -> None:
    cal = _train(tmp_path, "per_class", ScriptedLoop({("calibration", 0): 0.25}), name="platt")
    loaded = ThresholdedCalibration.from_dict(cal.to_dict())
    batch = [image("9", det(0.9, 0), det(0.3, 1), det(0.7, 2))]
    assert loaded.apply(batch) == cal.apply(batch)
    assert loaded.provenance == cal.provenance


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            {"scope": "global", "calibration_thresholds": {}, "operating_thresholds": {}},
            "calibrators",
        ),
        (
            {
                "scope": "x",
                "calibration_thresholds": {},
                "operating_thresholds": {},
                "calibrators": {},
            },
            "scope",
        ),
        (
            {"scope": "global", "operating_thresholds": {}, "calibrators": {}},
            "calibration_thresholds",
        ),
        (
            {
                "scope": "global",
                "calibration_thresholds": {},
                "operating_thresholds": {},
                "calibrators": {},
            },
            "exactly one calibrator",
        ),
    ],
)
def test_malformed_saved_calibrations(data: dict, message: str) -> None:
    with pytest.raises(CalibrationError, match=message):
        ThresholdedCalibration.from_dict(data)


def test_missing_thresholds_and_calibrators_are_errors() -> None:
    cal = ThresholdedCalibration("per_class", {0: 0.0}, {0: 0.0}, {})
    with pytest.raises(CalibrationError, match="no calibrator for class 0"):
        cal.calibrate([image("1", det(0.5, 0))])
    cal = ThresholdedCalibration("per_class", {}, {}, {})
    with pytest.raises(CalibrationError, match="no calibration threshold for class 0"):
        cal.calibrate([image("1", det(0.5, 0))])
    with pytest.raises(CalibrationError, match="no threshold for class 0"):
        apply_thresholds([image("1", det(0.5, 0))], {})


def test_train_refuses_an_unknown_scope(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="scope"):
        _train(tmp_path, "per_image", ScriptedLoop({}))


def test_keep_class_keeps_every_image() -> None:
    kept = keep_class([image("1", det(0.9, 0), det(0.9, 1)), image("2", det(0.1, 0))], 0, 0.5)
    assert [(i.image_id, len(i.detections)) for i in kept] == [("1", 1), ("2", 0)]
