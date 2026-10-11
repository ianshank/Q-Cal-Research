"""The experiment program: plans from cell factors, split roles, and one run end to end."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from qcal.config import ConfigError, load_config
from qcal.protocols import ImageDetections
from qcal.registry.experiments import ExperimentsError
from qcal_lab.calib.two_threshold import read_calibration
from qcal_lab.config import load_lab_config
from qcal_lab.data.coco import GroundTruth
from qcal_lab.evaluation import HandwrittenMissingError
from qcal_lab.experiment import (
    CALIBRATED_PREDICTIONS_FILE,
    CALIBRATION_FILE,
    RAW_PREDICTIONS_FILE,
    ROLES,
    PlanError,
    RoleSpec,
    RunRequest,
    resolve_plan,
    run_experiment,
    split_roles,
    write_result,
)
from qcal_lab.fixture_eval import FixtureEvalLoop
from qcal_lab.models import KIND_SCORE_DEFINITIONS
from qcal_lab.predictions import read_predictions
from tests.lab_support import fixture_project, lab_config

pytestmark = pytest.mark.rule("C3")

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
        ({"detector": "a", "quant_path": "ptq_minmax"}, "quant_path='ptq_minmax'"),
        ({"detector": "a", "fit_precision": "int8"}, "fit_precision='int8'"),
        ({"detector": "a", "split_design": "paper"}, "split_design='paper'"),
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
    with pytest.raises(PlanError, match="precision must be a non-empty string"):
        resolve_plan({"detector": "a"}, lab)


def test_every_pre_registered_axis_is_a_role_with_a_default(tmp_path: Path) -> None:
    plan = resolve_plan({"detector": "atss_r50"}, lab_config(tmp_path))
    assert (
        plan.target,
        plan.quant_path,
        plan.fit_precision,
        plan.threshold_regime,
        plan.split_design,
    ) == ("torch_fp32", "none", "fp32", "reuse_fp32", "disjoint")


def test_plan_roles_name_what_each_role_reads(tmp_path: Path) -> None:
    plan = resolve_plan(
        {"detector": "atss_r50", "precision": "fp32_tf32_off"}, lab_config(tmp_path)
    )
    spec = RoleSpec("id", "atss_r50", "fp32_tf32_off", "torch_fp32", "live")
    assert plan.roles() == dict.fromkeys(ROLES, spec)


@pytest.mark.parametrize("table", ["defaults", "supported"])
def test_a_factor_that_is_not_a_role_is_a_configuration_error(tmp_path: Path, table: str) -> None:
    """A validated but ignored factor would label runs with something they never ran."""
    value = '"x"' if table == "defaults" else '["x"]'
    lab = lab_config(tmp_path, f"[factors.{table}]\nbatch_size = {value}\n")
    with pytest.raises(ConfigError, match=r"\['batch_size'\], which are not roles"):
        resolve_plan({"detector": "a"}, lab)


@pytest.mark.parametrize("role", ["target", "quant_path", "fit_precision", "split_design"])
def test_role_labels_must_be_non_empty_strings(tmp_path: Path, role: str) -> None:
    lab = lab_config(tmp_path, f'[factors.supported]\n{role} = [""]\n')
    with pytest.raises(PlanError, match=f"{role} must be a non-empty string"):
        resolve_plan({"detector": "a", role: ""}, lab)


def test_artifact_file_names_are_per_split_templates() -> None:
    assert RAW_PREDICTIONS_FILE.format(split="val") == "raw_val.jsonl"
    assert CALIBRATED_PREDICTIONS_FILE.format(split="test") == "calibrated_test.jsonl"
    assert "{" not in CALIBRATION_FILE


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
        ('[splits]\nselect = "calibrator_fit_split"\n', "both fit calibrators and select"),
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
        self.seen: dict[str, set[str]] = {}

    def _record(
        self, purpose: str, predictions: Sequence[ImageDetections], gt: GroundTruth
    ) -> None:
        # Predictions and ground truth must always describe the same images.
        assert {i.image_id for i in predictions} == set(gt.images), purpose
        self.seen.setdefault(purpose, set()).update(gt.images)

    def targets(self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth) -> Any:
        self._record("targets", predictions, ground_truth)
        return self.inner.targets(predictions, ground_truth)

    def threshold_objective(self, predictions: Any, ground_truth: GroundTruth, **kw: Any) -> float:
        self._record(f"objective:{kw['stage']}", predictions, ground_truth)
        return self.inner.threshold_objective(predictions, ground_truth, **kw)

    def metrics(self, predictions: Any, ground_truth: GroundTruth) -> dict[str, float]:
        self._record("metrics", predictions, ground_truth)
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
    assert env["eval_loop"]["module"] == "qcal_lab.fixture_eval:build"
    assert env["eval_loop"]["file"].endswith("qcal_lab/fixture_eval.py")
    assert env["splits"]["evaluate"]["split"] == "test"
    assert env["fit_draw"] == {
        "seeded": True,
        "size": 6,
        "seed": 0,
        "sha256": env["fit_draw"]["sha256"],
    }
    for key in ("lab_config_sha256", "lab_config_effective_sha256", "qcal_lab_source_sha256"):
        assert len(env[key]) == 64, key
    assert env["calibration"]["provenance"]["calibration_threshold_split"] == "val"
    assert env["calibration"]["provenance"]["operating_threshold_split"] == "val"
    assert env["calibration"]["provenance"]["fit_split"] == "calibrator_fit_split"
    retained = env["evaluate_retained_detections"]
    assert retained["total"] == sum(len(i.detections) for i in images)
    assert set(env["prediction_cache"]) == {"calibrator_fit_split", "val", "test"}
    assert env["program_python"]
    assert env["detector_runtime"] == {}  # the fixture detector has no backend switches

    write_result(root / "out/result.json", result)
    assert (root / "out/result.json").read_text().startswith("{")


def test_the_evaluate_split_is_only_ever_used_for_metrics(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    spy = SpyLoop()
    run_experiment(_request(root, cells[4]), eval_loop=spy)  # isotonic, per class
    # Calibrator targets come from the fit split, both thresholds from val (CLAUDE.md rule 3),
    # and the test split is used for metrics only.
    assert spy.seen["targets"] <= _manifest(root, "calibrator_fit_split")
    assert spy.seen["objective:calibration"] == _manifest(root, "val")
    assert spy.seen["objective:operating"] == _manifest(root, "val")
    assert spy.seen["metrics"] == _manifest(root, "test")


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
    assert all(not use["hit"] for use in first.environment["prediction_cache"].values())
    assert second.environment["prediction_cache"]["val"] == {"hit": True, "produced_by": "R-a"}


def test_a_cache_entry_that_does_not_match_the_run_is_recomputed(
    project: tuple[Path, list[str]],
) -> None:
    import hashlib
    import json

    root, cells = project
    first = run_experiment(_request(root, cells[0], run_id="R-a"))
    path = root / next(a["path"] for a in first.artifacts if a["kind"] == "predictions_raw_val")
    # A forged entry whose metadata agrees with its bytes, but which holds other images.
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n" + lines[1].replace('"image_id":"', '"image_id":"x') + "\n")
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta))
    second = run_experiment(_request(root, cells[0], run_id="R-b"))
    assert second.environment["prediction_cache"]["val"] == {"hit": False, "produced_by": "R-b"}
    assert second.metrics == first.metrics


def test_a_corrupt_cache_sidecar_is_ignored(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    first = run_experiment(_request(root, cells[0], run_id="R-a"))
    path = root / next(a["path"] for a in first.artifacts if a["kind"] == "predictions_raw_val")
    path.with_suffix(".meta.json").write_text("{")
    second = run_experiment(_request(root, cells[0], run_id="R-b"))
    assert second.environment["prediction_cache"]["val"]["hit"] is False


def test_without_a_cache_raw_predictions_go_next_to_the_run(
    project: tuple[Path, list[str]],
) -> None:
    root, cells = project
    with (root / "configs/lab.toml").open("a") as handle:
        handle.write('[predictions]\ncache_dir = ""\n')
    result = run_experiment(_request(root, cells[0]))
    raw = [a["path"] for a in result.artifacts if a["kind"].startswith("predictions_raw")]
    assert raw == [
        "runs/results/R-test/raw_calibrator_fit_split.jsonl",
        "runs/results/R-test/raw_val.jsonl",
        "runs/results/R-test/raw_test.jsonl",
    ]


def test_overlapping_manifests_are_refused_before_detector_work(
    project: tuple[Path, list[str]],
) -> None:
    root, cells = project
    shared = min(_manifest(root, "test"))
    with (root / "data/manifests/val.txt").open("a") as handle:
        handle.write(shared + "\n")
    with pytest.raises(PlanError, match="fail the leakage check"):
        run_experiment(_request(root, cells[0]))
    assert not (root / "runs/cache").exists()


def test_roles_sharing_images_are_refused(project: tuple[Path, list[str]]) -> None:
    from qcal_lab.experiment import check_disjoint

    root, _ = project
    config = load_config(root, environ={})
    ids = {"fit": ("1", "2"), "select": ("3",), "evaluate": ("2",)}
    with pytest.raises(PlanError, match="fit and evaluate splits share 1 image"):
        check_disjoint(config, ids)


def test_a_missing_manifest_is_refused(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    (root / "data/manifests/trt_calib_images.txt").unlink()
    with pytest.raises(PlanError, match="missing split: trt_calib_images"):
        run_experiment(_request(root, cells[0]))


def test_a_non_ian_evaluation_loop_cannot_report_metrics(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    text = (root / "configs/lab.toml").read_text().replace("qcal_lab.fixture_eval", "json")
    (root / "configs/lab.toml").write_text(text)
    with pytest.raises(ConfigError, match="not one of Ian's hand-written files"):
        run_experiment(_request(root, cells[0]))


def test_injected_loops_are_recorded_as_injected(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    result = run_experiment(_request(root, cells[0]), eval_loop=FixtureEvalLoop())
    assert result.environment["eval_loop"] == {"module": "injected:FixtureEvalLoop"}


def test_a_target_the_detector_does_not_produce_is_refused_before_detector_work(
    project: tuple[Path, list[str]],
) -> None:
    """Supporting a target in configuration does not make a detector produce it."""
    root, _ = project
    lab_file = root / "configs/lab.toml"
    lab_file.write_text(
        lab_file.read_text() + '[factors.supported]\ntarget = ["torch_fp32", "trt_jetson"]\n'
    )
    experiments = yaml.safe_load((root / "EXPERIMENTS.yaml").read_text())
    experiments["cells"].append({"id": "C-jetson", "detector": "fixture", "target": "trt_jetson"})
    (root / "EXPERIMENTS.yaml").write_text(yaml.safe_dump(experiments, sort_keys=False))
    with pytest.raises(PlanError, match="does not produce target 'trt_jetson'"):
        run_experiment(_request(root, "C-jetson"))
    assert not (root / "runs/cache").exists()


def test_a_run_records_each_roles_source(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    roles = run_experiment(_request(root, cells[0])).environment["roles"]
    assert set(roles) == set(ROLES)
    assert roles["fit"] == {
        "dataset": "id",
        "detector": "fixture",
        "precision": "fp32",
        "target": "torch_fp32",
        "source": "live",
    }


def test_a_run_records_what_produced_its_predictions(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    result = run_experiment(_request(root, cells[0]))
    source = result.environment["predictions_source"]
    assert source == {
        "precision": "fp32",
        "target": "torch_fp32",
        "quant_path": "none",
        "shift": "id",
        "score_definition": KIND_SCORE_DEFINITIONS["fixture"],
        "model_sha256": "",  # the fixture detector has no checkpoint
        "test_cfg": {},
    }
    for artifact in result.artifacts:
        if artifact["path"].endswith(".jsonl"):
            header, _ = read_predictions(root / artifact["path"])
            assert header.source.to_dict() == source, artifact["kind"]
    calibration = next(a for a in result.artifacts if a["kind"] == "calibration")
    read_calibration(root / calibration["path"])  # the saved calibration carries its envelope


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


def test_changed_image_bytes_invalidate_cached_predictions(
    project: tuple[Path, list[str]],
) -> None:
    root, cells = project
    text = (root / "configs/lab.toml").read_text().replace('images_dir = ""', 'images_dir = "imgs"')
    (root / "configs/lab.toml").write_text(text)
    (root / "imgs").mkdir()
    val_image = min(_manifest(root, "val"))
    image_file = root / "imgs" / f"fixture_{int(val_image):04d}.jpg"
    image_file.write_bytes(b"v1")
    first = run_experiment(_request(root, cells[0], run_id="R-a"))
    assert first.environment["prediction_cache"]["val"]["hit"] is False
    assert run_experiment(_request(root, cells[0], run_id="R-b")).environment["prediction_cache"][
        "val"
    ]["hit"]
    image_file.write_bytes(b"v2")  # same path, different pixels
    third = run_experiment(_request(root, cells[0], run_id="R-c")).environment
    assert third["prediction_cache"]["val"] == {"hit": False, "produced_by": "R-c"}
    assert third["prediction_cache"]["test"]["hit"] is True  # other splits are unaffected
