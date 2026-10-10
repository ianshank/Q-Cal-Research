"""The experiment program: plans from cell factors, split roles, and one run end to end."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from qcal.config import ConfigError, load_config
from qcal.protocols import ImageDetections
from qcal.registry.experiments import ExperimentsError
from qcal_lab.config import load_lab_config
from qcal_lab.data.coco import GroundTruth
from qcal_lab.evaluation import HandwrittenMissingError
from qcal_lab.experiment import (
    PlanError,
    RunRequest,
    resolve_plan,
    run_experiment,
    split_roles,
    write_result,
)
from qcal_lab.fixture_eval import FixtureEvalLoop
from qcal_lab.predictions import read_predictions
from tests.lab_support import fixture_project, lab_config

# --- plans -------------------------------------------------------------------------------------


def test_plan_defaults(tmp_path: Path) -> None:
    plan = resolve_plan({"detector": "atss_r50"}, lab_config(tmp_path))
    assert (plan.calibrator, plan.scope, plan.shift, plan.precision, plan.fit_size) == (
        "none",
        "per_class",
        "id",
        "fp32",
        None,
    )


def test_plan_reads_every_role(tmp_path: Path) -> None:
    factors = {
        "detector": "atss_r50",
        "calibrator": "isotonic",
        "calibrator_scope": "global",
        "calibrator_fit_split_size": 500,
        "precision": "fp32_tf32_off",
        "target": "torch_fp32",
        "threshold_regime": "reuse_fp32",
        "fit_precision": "fp32",
        "note": "anything",
    }
    plan = resolve_plan(factors, lab_config(tmp_path))
    assert (plan.calibrator, plan.scope, plan.fit_size, plan.precision) == (
        "isotonic",
        "global",
        500,
        "fp32_tf32_off",
    )


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        ({"detector": "a", "learning_rate": 0.1}, "does not understand: learning_rate"),
        ({"detector": "a", "precision": "ptq_entropy"}, "precision='ptq_entropy' is not supported"),
        ({"detector": "a", "target": "trt_jetson"}, "target='trt_jetson'"),
        ({"detector": "a", "threshold_regime": "reselect_int8"}, "threshold_regime"),
        ({"detector": "a", "calibrator": "temperature"}, "unknown calibrator"),
        ({"detector": "a", "calibrator_scope": "image"}, "calibrator_scope='image'"),
        ({"detector": "a", "calibrator_fit_split_size": 0}, "positive integer"),
        ({"detector": "a", "calibrator_fit_split_size": True}, "positive integer"),
        ({"detector": "a", "shift": "cocoC_s3"}, "no datasets.cocoC_s3 entry"),
        ({"calibrator": "none"}, "must name a 'detector'"),
        ({"detector": ""}, "must name a 'detector'"),
    ],
)
def test_plans_refuse_what_the_program_would_not_honour(
    tmp_path: Path, factors: dict[str, Any], message: str
) -> None:
    with pytest.raises(PlanError, match=message):
        resolve_plan(factors, lab_config(tmp_path))


def test_plan_configuration_errors(tmp_path: Path) -> None:
    lab = lab_config(tmp_path, '[factors.supported]\nprecision = "fp32"\n')
    with pytest.raises(ConfigError, match="must be a list"):
        resolve_plan({"detector": "a"}, lab)
    lab = lab_config(
        tmp_path, "[factors.defaults]\nprecision = 32\n[factors.supported]\nprecision = [32]\n"
    )
    with pytest.raises(PlanError, match="precision must be a string"):
        resolve_plan({"detector": "a"}, lab)


# --- split roles ------------------------------------------------------------------------------


def test_split_roles(repo: Path, tmp_path: Path) -> None:
    config = load_config(repo, environ={})
    assert split_roles(lab_config(tmp_path), config) == {
        "fit": "calibrator_fit_split",
        "select": "val",
        "evaluate": "test",
    }


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[splits]\nselect = "test"\n', "tuning on the test split"),
        ('[splits]\nfit = "test"\n', "tuning on the test split"),
        ('[splits]\nfit = "train"\n', "not in data.splits"),
        ('[calibration]\noperating_threshold_role = "evaluate"\n', "must be one of"),
    ],
)
def test_split_roles_refuse_test_reuse(repo: Path, tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        split_roles(lab_config(tmp_path, body), load_config(repo, environ={}))


# --- one run ----------------------------------------------------------------------------------


class SpyLoop:
    """Records which images the pipeline passed to each method, by purpose.

    It wraps (rather than subclasses) the fixture loop, so the loop's own internal calls are
    not mistaken for calls from the pipeline.
    """

    def __init__(self) -> None:
        self.inner = FixtureEvalLoop()
        self.seen: dict[str, set[str]] = {"targets": set(), "objective": set(), "metrics": set()}

    def targets(self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth) -> Any:
        self.seen["targets"] |= set(ground_truth.images)
        return self.inner.targets(predictions, ground_truth)

    def threshold_objective(self, predictions: Any, ground_truth: GroundTruth, **kw: Any) -> float:
        self.seen["objective"] |= set(ground_truth.images)
        return self.inner.threshold_objective(predictions, ground_truth, **kw)

    def metrics(self, predictions: Any, ground_truth: GroundTruth) -> dict[str, float]:
        self.seen["metrics"] |= set(ground_truth.images)
        return self.inner.metrics(predictions, ground_truth)


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, list[str]]:
    return fixture_project(tmp_path)


def _request(project: Path, cell: str, seed: int = 0, run_id: str = "R-test") -> RunRequest:
    return RunRequest(project, run_id, cell, seed, project / "runs/results" / f"{run_id}.json")


def _manifest(project: Path, split: str) -> set[str]:
    lines = (project / f"data/manifests/{split}.txt").read_text().splitlines()
    return {line for line in lines if line and not line.startswith("#")}


def test_a_run_produces_metrics_artifacts_and_provenance(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    result = run_experiment(_request(root, cells[2]))  # platt, per class
    assert set(result.metrics) == {"smoke_images", "smoke_detections", "smoke_mean_abs_gap"}
    kinds = [a["kind"] for a in result.artifacts]
    assert kinds == [
        "predictions_raw_calibrator_fit_split",
        "predictions_raw_val",
        "predictions_raw_test",
        "calibration",
        "predictions_calibrated_test",
    ]
    assert all((root / a["path"]).is_file() for a in result.artifacts)
    header, images = read_predictions(root / result.artifacts[-1]["path"])
    assert header.stage == "calibrated"
    assert {i.image_id for i in images} == _manifest(root, "test")
    env = result.environment
    assert env["eval_loop"] == "qcal_lab.fixture_eval:build"
    assert env["splits"]["evaluate"]["split"] == "test"
    assert env["fit_draw"] == {"size": 6, "seed": 0, "sha256": env["fit_draw"]["sha256"]}
    assert len(env["lab_config_sha256"]) == 64

    write_result(root / "out/result.json", result)
    assert (root / "out/result.json").read_text().startswith("{")


def test_the_evaluate_split_is_only_ever_used_for_metrics(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    spy = SpyLoop()
    run_experiment(_request(root, cells[4]), eval_loop=spy)  # isotonic, per class
    test_ids = _manifest(root, "test")
    assert spy.seen["metrics"] == test_ids
    assert not (spy.seen["targets"] | spy.seen["objective"]) & test_ids
    assert spy.seen["targets"] <= _manifest(root, "calibrator_fit_split")


def test_seeds_draw_different_fit_subsets(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    text = (root / "EXPERIMENTS.yaml").read_text().replace("seeds:\n- 0\n", "seeds:\n- 0\n- 1\n")
    (root / "EXPERIMENTS.yaml").write_text(text)
    first = run_experiment(_request(root, cells[0], 0, "R-a")).environment["fit_draw"]
    second = run_experiment(_request(root, cells[0], 1, "R-b")).environment["fit_draw"]
    assert first["sha256"] != second["sha256"]


def test_raw_predictions_are_cached_and_reused(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    first = run_experiment(_request(root, cells[0], run_id="R-a"))
    second = run_experiment(_request(root, cells[2], run_id="R-b"))
    raw = [a["path"] for a in first.artifacts if a["kind"].startswith("predictions_raw")]
    assert raw == [a["path"] for a in second.artifacts if a["kind"].startswith("predictions_raw")]
    assert all(p.startswith("runs/cache/predictions/") for p in raw)


def test_without_a_cache_raw_predictions_go_next_to_the_run(
    project: tuple[Path, list[str]],
) -> None:
    root, cells = project
    with (root / "configs/lab.toml").open("a") as handle:
        handle.write('[predictions]\ncache_dir = ""\n[splits]\nselect = "calibrator_fit_split"\n')
    result = run_experiment(_request(root, cells[0]))
    raw = [a["path"] for a in result.artifacts if a["kind"].startswith("predictions_raw")]
    # fit and select share one split: it runs once and is recorded once.
    assert raw == [
        "runs/results/R-test/raw_calibrator_fit_split.jsonl",
        "runs/results/R-test/raw_test.jsonl",
    ]


def test_unregistered_seed_or_cell_is_refused(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    with pytest.raises(PlanError, match="seed 7 is not pre-registered"):
        run_experiment(_request(root, cells[0], seed=7))
    with pytest.raises(ExperimentsError, match="not pre-registered"):
        run_experiment(_request(root, "C-unknown"))


def test_a_missing_handwritten_loop_fails_before_any_detector_work(
    project: tuple[Path, list[str]],
) -> None:
    root, cells = project
    (root / "configs/lab.toml").write_text(
        '[datasets.id]\nannotations = "data/fixture/annotations.json"\nimages_dir = ""\n'
    )
    with pytest.raises(HandwrittenMissingError):
        run_experiment(_request(root, cells[0]))
    assert not (root / "runs/cache").exists()


def test_a_detector_that_drops_images_is_refused(
    project: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cells = project
    from qcal_lab.models.fixture import FixtureDetector

    monkeypatch.setattr(FixtureDetector, "predict", lambda self, ids: [])
    with pytest.raises(PlanError, match="out of order or incomplete"):
        run_experiment(_request(root, cells[0]))


def test_the_repository_lab_config_must_be_hashed(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    lab = load_lab_config(root, root / "configs/lab.toml")
    (root / "elsewhere.toml").write_text((root / "configs/lab.toml").read_text())
    outside = load_lab_config(root, root / "elsewhere.toml")
    assert lab.path is not None
    with pytest.raises(ConfigError, match="not covered"):
        run_experiment(_request(root, cells[0]), lab=outside)
