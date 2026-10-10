"""The seam to Ian's evaluation loop, and the fixture-only stand-in."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from qcal.config import ConfigError
from qcal_lab.data.fixture import NotFixtureError
from qcal_lab.evaluation import (
    EvalLoop,
    EvaluationError,
    HandwrittenMissingError,
    check_metrics,
    check_targets,
    load_eval_loop,
)
from qcal_lab.fixture_eval import FixtureEvalLoop, box_iou
from tests.lab_support import det, ground_truth, image, lab_config


def _lab(tmp_path: Path, module: str, factory: str = "build"):
    return lab_config(tmp_path, f'[eval_loop]\nmodule = "{module}"\nfactory = "{factory}"\n')


def test_the_default_is_ians_handwritten_module_and_it_is_missing(tmp_path: Path) -> None:
    lab = lab_config(tmp_path)
    assert lab.text("eval_loop.module") == "qcal_lab.handwritten.eval_loop"
    with pytest.raises(HandwrittenMissingError, match="Ian's to write by hand"):
        load_eval_loop(lab)


@pytest.fixture
def module_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "mods"
    directory.mkdir()
    monkeypatch.syspath_prepend(str(directory))
    yield directory
    for name in [m for m in sys.modules if m.startswith("labtest_")]:
        del sys.modules[name]


def test_a_missing_dependency_inside_the_loop_is_not_hidden(
    module_dir: Path, tmp_path: Path
) -> None:
    (module_dir / "labtest_broken.py").write_text("import labtest_no_such_dependency\n")
    with pytest.raises(ModuleNotFoundError, match="labtest_no_such_dependency"):
        load_eval_loop(_lab(tmp_path, "labtest_broken"))


def test_factory_must_exist_and_return_a_loop(module_dir: Path, tmp_path: Path) -> None:
    (module_dir / "labtest_loops.py").write_text(
        "build = 3\n"
        "def wrong(options):\n    return object()\n"
        "def good(options):\n    from qcal_lab.fixture_eval import FixtureEvalLoop\n"
        "    assert options == {'k': 1}\n    return FixtureEvalLoop()\n"
    )
    with pytest.raises(ConfigError, match="no callable 'build'"):
        load_eval_loop(_lab(tmp_path, "labtest_loops"))
    with pytest.raises(ConfigError, match="must return an object"):
        load_eval_loop(_lab(tmp_path, "labtest_loops", "wrong"))
    lab = lab_config(
        tmp_path,
        '[eval_loop]\nmodule = "labtest_loops"\nfactory = "good"\n[eval_loop.options]\nk = 1\n',
    )
    assert isinstance(load_eval_loop(lab), EvalLoop)


def test_check_targets() -> None:
    preds = [image("1", det(0.5), det(0.4))]
    assert check_targets(preds, {"1": [0.1, 1]}) == {"1": (0.1, 1.0)}
    for bad, message in [
        ({}, "expected 2 targets"),
        ({"1": [0.1]}, "expected 2 targets"),
        ({"1": [0.1, "x"]}, "not a number"),
        ({"1": [0.1, True]}, "not a number"),
        ({"1": [0.1, 1.5]}, "not in"),
        ({"1": [0.1, float("inf")]}, "not in"),
    ]:
        with pytest.raises(EvaluationError, match=message):
            check_targets(preds, bad)


def test_check_metrics() -> None:
    assert check_metrics({"LaECE0": 7, "AP": 41.5}) == {"LaECE0": 7.0, "AP": 41.5}
    for bad, message in [
        ({}, "no metrics"),
        ({"La ECE": 1.0}, "not usable"),
        ({"AP": True}, "not a number"),
        ({"AP": float("nan")}, "not finite"),
    ]:
        with pytest.raises(EvaluationError, match=message):
            check_metrics(bad)


# --- the fixture stand-in ----------------------------------------------------------------------


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_fixture_loop_refuses_real_datasets() -> None:
    loop = FixtureEvalLoop()
    with pytest.raises(NotFixtureError):
        loop.targets([], ground_truth())
    with pytest.raises(NotFixtureError):
        loop.metrics([], ground_truth())


def test_fixture_loop_values() -> None:
    gt = ground_truth(fixture=True, boxes={1: [(1, [0.0, 0.0, 10.0, 10.0])]})
    loop = FixtureEvalLoop()
    preds = [image("1", det(0.9, 0, (0, 0, 10, 10)), det(0.4, 0, (50, 50, 60, 60))), image("2")]
    assert loop.targets(preds, gt) == {"1": [1.0, 0.0], "2": []}
    # One poor detection out of two, plus the object it found: (1 + 0) / (2 + 1).
    assert loop.threshold_objective(preds, gt, label=0, stage="calibration") == pytest.approx(1 / 3)
    assert loop.threshold_objective([image("2")], gt, label=1, stage="operating") != 0.0
    metrics = loop.metrics(preds, gt)
    assert set(metrics) == {"smoke_images", "smoke_detections", "smoke_mean_abs_gap"}
    assert metrics["smoke_mean_abs_gap"] == pytest.approx((0.1 + 0.4) / 2)
    assert loop.metrics([image("2")], gt)["smoke_mean_abs_gap"] == 0.0
