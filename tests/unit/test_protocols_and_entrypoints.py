"""Protocol shapes for Phase 1 components, and the ``python -m`` entry points."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from qcal.protocols import Calibrator, Detection, Detector, ImageDetections, Metric, Quantizer


class _Detector:
    name = "fake"

    def predict(self, image_ids: Sequence[str]) -> Sequence[ImageDetections]:
        box = (0.0, 0.0, 1.0, 1.0)
        return [ImageDetections(i, (Detection(box, 0.9, 1, logit=2.2),)) for i in image_ids]


class _Calibrator:
    name = "identity"

    def fit(self, scores: Sequence[float], targets: Sequence[float]) -> None:
        self.fitted = len(scores) == len(targets)

    def transform(self, scores: Sequence[float]) -> Sequence[float]:
        return list(scores)


class _Metric:
    name = "count"

    def compute(
        self, predictions: Sequence[ImageDetections], ground_truth: Mapping[str, Any]
    ) -> float:
        return float(sum(len(p.detections) for p in predictions))


class _Quantizer:
    name = "noop"

    def build(
        self, model_path: Path, calibration_images: Sequence[str], output: Path
    ) -> Mapping[str, Any]:
        return {"images": len(calibration_images), "output": str(output)}


def test_fakes_satisfy_the_runtime_protocols() -> None:
    assert isinstance(_Detector(), Detector)
    assert isinstance(_Calibrator(), Calibrator)
    assert isinstance(_Metric(), Metric)
    assert isinstance(_Quantizer(), Quantizer)
    assert not isinstance(object(), Detector)


def test_protocol_data_shapes_are_immutable() -> None:
    detections = _Detector().predict(["img1"])
    assert _Metric().compute(detections, {}) == 1.0
    with pytest.raises(AttributeError):
        detections[0].image_id = "other"  # type: ignore[misc]


def test_python_dash_m_entry_points(repo: Path) -> None:
    version = subprocess.run(
        [sys.executable, "-m", "qcal", "--version"], capture_output=True, text=True, check=True
    )
    assert version.stdout.startswith("qcal ")
    payload = json.dumps(
        {"tool_name": "Write", "tool_input": {"file_path": str(repo / "CLAIMS.md")}}
    )
    hook = subprocess.run(
        [sys.executable, "-m", "qcal.hooks", "guard-paths"],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        env={"CLAUDE_PROJECT_DIR": str(repo), "PATH": "/usr/bin:/bin"},
    )
    assert hook.returncode == 2
    assert "Ian-only" in hook.stderr


def test_hook_module_main_is_importable() -> None:
    from qcal.hooks import cli

    err = io.StringIO()
    assert cli.main(["guard-bash"], stdin=io.StringIO("{}"), stderr=err, environ={}) == 0
