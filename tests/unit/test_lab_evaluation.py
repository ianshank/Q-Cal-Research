"""The seam to Ian's evaluation loop, and the fixture-only stand-in."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from qcal.config import Config, ConfigError, load_config
from qcal_lab.data.fixture import NotFixtureError
from qcal_lab.evaluation import (
    EvalLoop,
    EvaluationError,
    HandwrittenMissingError,
    check_metrics,
    check_targets,
    load_eval_loop,
)
from qcal_lab.fixture_eval import (
    FixtureEvalLoop,
    best_iou_targets,
    box_iou,
    poor_or_missed_share,
    smoke_metrics,
)
from tests.lab_support import det, fixture_ground_truth, ground_truth, image, lab_config


def _lab(root: Path, module: str, factory: str = "build", options: str = ""):
    return lab_config(root, f'[eval_loop]\nmodule = "{module}"\nfactory = "{factory}"\n{options}')


def test_the_default_is_ians_handwritten_module_and_it_is_missing(config: Config) -> None:
    lab = lab_config(config.root)
    assert lab.text("eval_loop.module") == "qcal_lab.handwritten.eval_loop"
    with pytest.raises(HandwrittenMissingError, match="Ian's to write by hand"):
        load_eval_loop(lab, config)


@pytest.fixture
def handwritten(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An importable directory inside the repository that policy classes as Ian-only."""
    directory = repo / "lab" / "handwritten"
    directory.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(directory))
    yield directory
    for name in [m for m in sys.modules if m.startswith("labtest_")]:
        del sys.modules[name]


def test_a_missing_dependency_inside_the_loop_is_not_hidden(
    handwritten: Path, config: Config
) -> None:
    (handwritten / "labtest_broken.py").write_text("import labtest_no_such_dependency\n")
    with pytest.raises(ModuleNotFoundError, match="labtest_no_such_dependency"):
        load_eval_loop(_lab(config.root, "labtest_broken"), config)


def test_factory_must_exist_and_return_a_loop(handwritten: Path, config: Config) -> None:
    (handwritten / "labtest_loops.py").write_text(
        "build = 3\n"
        "def wrong(options):\n    return object()\n"
        "def good(options):\n    from qcal_lab.fixture_eval import FixtureEvalLoop\n"
        "    assert options == {'k': 1}\n    return FixtureEvalLoop()\n"
    )
    with pytest.raises(ConfigError, match="no callable 'build'"):
        load_eval_loop(_lab(config.root, "labtest_loops"), config)
    with pytest.raises(ConfigError, match="must return an object"):
        load_eval_loop(_lab(config.root, "labtest_loops", "wrong"), config)
    lab = _lab(config.root, "labtest_loops", "good", "[eval_loop.options]\nk = 1\n")
    loaded = load_eval_loop(lab, config)
    assert isinstance(loaded.loop, EvalLoop)
    assert loaded.provenance()["file"] == "lab/handwritten/labtest_loops.py"
    assert loaded.module == "labtest_loops:good"
    assert len(loaded.sha256) == 64


def test_only_ian_only_files_or_the_fixture_stand_in_may_report_metrics(
    repo: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_dir = repo / "lab" / "agent_code"
    agent_dir.mkdir(parents=True)
    (agent_dir / "labtest_fake.py").write_text(
        "from qcal_lab.fixture_eval import build  # noqa: F401\n"
    )
    monkeypatch.syspath_prepend(str(agent_dir))
    try:
        with pytest.raises(ConfigError, match="not one of Ian's hand-written files"):
            load_eval_loop(_lab(config.root, "labtest_fake"), config)
        with pytest.raises(ConfigError, match="not one of Ian's hand-written files"):
            load_eval_loop(_lab(config.root, "json"), config)  # outside the repository
    finally:
        sys.modules.pop("labtest_fake", None)
    loaded = load_eval_loop(_lab(config.root, "qcal_lab.fixture_eval"), config)
    assert isinstance(loaded.loop, FixtureEvalLoop)


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


def test_stand_in_computations() -> None:
    gt = ground_truth(boxes={1: [(1, [0.0, 0.0, 10.0, 10.0])]})
    preds = [image("1", det(0.9, 0, (0, 0, 10, 10)), det(0.4, 0, (50, 50, 60, 60))), image("2")]
    assert best_iou_targets(preds, gt) == {"1": [1.0, 0.0], "2": []}
    # One poor detection out of two, plus the object it found: (1 + 0) / (2 + 1).
    assert poor_or_missed_share(preds, gt, 0) == pytest.approx(1 / 3)
    assert math.isnan(poor_or_missed_share([image("2")], gt, 1))  # neither detections nor objects
    metrics = smoke_metrics(preds, gt)
    assert set(metrics) == {"smoke_images", "smoke_detections", "smoke_mean_abs_gap"}
    assert metrics["smoke_mean_abs_gap"] == pytest.approx((0.1 + 0.4) / 2)
    assert smoke_metrics([image("2")], gt)["smoke_mean_abs_gap"] == 0.0


def test_fixture_loop_runs_on_a_generated_fixture_only() -> None:
    loop = FixtureEvalLoop()
    fixture = fixture_ground_truth()
    preds = [image("1", det(0.5, 0, (0, 0, 10, 10)))]
    assert loop.targets(preds, fixture) == best_iou_targets(preds, fixture)
    assert loop.metrics(preds, fixture) == smoke_metrics(preds, fixture)
    loop.threshold_objective(preds, fixture, label=0, stage="operating")
    real = ground_truth()
    for call in (
        lambda: loop.targets([], real),
        lambda: loop.metrics([], real),
        lambda: loop.threshold_objective([], real, label=0, stage="calibration"),
    ):
        with pytest.raises(NotFixtureError):
            call()


def test_load_eval_loop_needs_a_source_file(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    module = types.ModuleType("labtest_sourceless")
    monkeypatch.setitem(sys.modules, "labtest_sourceless", module)
    with pytest.raises(ConfigError, match="no source file"):
        load_eval_loop(_lab(config.root, "labtest_sourceless"), config)


def test_repository_config_classes_handwritten_as_ian_only() -> None:
    from qcal.policy import Policy
    from tests.conftest import REPO_ROOT

    policy = Policy.from_config(load_config(REPO_ROOT, environ={}))
    assert "ian_only" in policy.categories_for("src/qcal_lab/handwritten/eval_loop.py")
