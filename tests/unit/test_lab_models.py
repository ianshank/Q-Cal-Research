"""Detectors: the registry, the fixture detector and the MMDetection adapter (with fakes)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qcal.config import ConfigError
from qcal.protocols import Detector
from qcal_lab.data.coco import parse_coco
from qcal_lab.data.fixture import NotFixtureError, build_fixture, fixture_bytes, is_fixture
from qcal_lab.models import (
    DETECTORS,
    KIND_TARGETS,
    DetectorError,
    build_detector,
    detector_spec,
    detector_targets,
)
from qcal_lab.models import mmdet as mmdet_module
from qcal_lab.models.base import setting_int, setting_number
from qcal_lab.models.fixture import FixtureDetector
from tests.lab_support import fixture_document, fixture_ground_truth, ground_truth, lab_config

FIXTURE = fixture_ground_truth()


# --- fixture dataset and detector ------------------------------------------------------------


def test_fixture_dataset_is_deterministic_and_marked() -> None:
    kwargs = {"images": 4, "categories": 2, "max_objects_per_image": 2, "width": 8, "height": 8}
    assert build_fixture(seed=1, **kwargs) == build_fixture(seed=1, **kwargs)
    assert build_fixture(seed=1, **kwargs) != build_fixture(seed=2, **kwargs)
    assert is_fixture(FIXTURE)
    assert is_fixture(FIXTURE.subset(["1", "2"]))
    assert not is_fixture(ground_truth())
    with pytest.raises(ValueError, match="positive"):
        build_fixture(seed=1, **{**kwargs, "images": 0})


def _digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(fixture_bytes(document)).hexdigest()


def test_the_fixture_is_recognised_by_content_not_by_its_label() -> None:
    """A copied description, edited boxes or forged parameters do not make a fixture."""
    spoofed = ground_truth(description=fixture_document()["info"]["description"])
    assert not is_fixture(spoofed)
    edited = fixture_document()
    edited["annotations"][0]["bbox"] = [0.0, 0.0, 1.0, 1.0]
    assert not is_fixture(parse_coco(edited, sha256=_digest(edited)))
    forged = fixture_document()
    forged["info"]["fixture_parameters"] = {"images": "many"}
    assert not is_fixture(parse_coco(forged, sha256=_digest(forged)))
    missing = fixture_document()
    del missing["info"]["fixture_parameters"]
    assert not is_fixture(parse_coco(missing, sha256=_digest(missing)))


def test_real_detectors_never_see_ground_truth_boxes(tmp_path: Path) -> None:
    from qcal_lab.models import DETECTORS

    seen = {}

    def spy(context: Any) -> Any:
        seen["boxes"] = sum(len(b) for b in context.ground_truth.boxes.values())
        seen["images"] = len(context.ground_truth.images)
        return SimpleNamespace(name="spy", predict=lambda ids: [])

    DETECTORS.register("labtest_spy", spy)
    try:
        lab = lab_config(tmp_path, '[detectors.spy]\nkind = "labtest_spy"\n')
        build_detector(lab, "spy", ground_truth(), precision="fp32")
        assert seen == {"boxes": 0, "images": 3}
        build_detector(lab, "spy", FIXTURE, precision="fp32")
        assert seen["boxes"] > 0  # only the synthetic fixture's stand-in gets boxes
    finally:
        DETECTORS._items.pop("labtest_spy")


def test_fixture_detector_is_deterministic_per_image(tmp_path: Path) -> None:
    detector = build_detector(lab_config(tmp_path), "fixture", FIXTURE, precision="fp32")
    assert isinstance(detector, Detector)
    ids = list(FIXTURE.image_ids())
    forward = detector.predict(ids)
    backward = detector.predict(list(reversed(ids)))
    assert forward == list(reversed(backward))
    for result in forward:
        scores = [d.score for d in result.detections]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert all(0 <= d.label < FIXTURE.num_classes for d in result.detections)
    with pytest.raises(DetectorError, match="not in the fixture"):
        detector.predict(["999"])


def test_fixture_detector_refuses_real_datasets(tmp_path: Path) -> None:
    with pytest.raises(NotFixtureError):
        build_detector(lab_config(tmp_path), "fixture", ground_truth(), precision="fp32")


def test_zero_jitter_keeps_boxes(tmp_path: Path) -> None:
    detector = FixtureDetector(
        "f",
        FIXTURE,
        seed=0,
        detect_probability=1.0,
        box_jitter=0.0,
        label_flip_probability=0.0,
        false_positives_per_image=0,
        overconfidence=1.0,
    )
    result = detector.predict(["1"])[0]
    assert sorted(d.box_xyxy for d in result.detections) == sorted(
        b.box_xyxy for b in FIXTURE.boxes["1"]
    )


def test_detector_configuration_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="'yolo' is not configured"):
        build_detector(lab_config(tmp_path), "yolo", FIXTURE, precision="fp32")
    with pytest.raises(ConfigError, match="kind must name"):
        build_detector(
            lab_config(tmp_path, "[detectors.x]\na = 1\n"), "x", FIXTURE, precision="fp32"
        )
    lab = lab_config(tmp_path, "[detectors.fixture]\nbox_jitter = 2.0\n")
    with pytest.raises(ConfigError, match=r"must lie in \[0.0, 1.0\]"):
        build_detector(lab, "fixture", FIXTURE, precision="fp32")
    with pytest.raises(ConfigError, match="must be a number"):
        setting_number({"k": "1"}, "k", low=0, high=1)
    with pytest.raises(ConfigError, match="non-negative integer"):
        setting_int({"k": -1}, "k")


def test_detector_spec_is_json_safe_and_complete(tmp_path: Path) -> None:
    spec = detector_spec(lab_config(tmp_path), "fixture", "fp32")
    assert spec["name"] == "fixture"
    assert spec["precision"] == "fp32"
    assert spec["settings"]["kind"] == "fixture"


# --- MMDetection adapter ----------------------------------------------------------------------


class FakeTensor:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[Any]:
        return self.values


class FakeApi:
    def __init__(self, rows: list[tuple[list[float], float, int]]) -> None:
        self.rows = rows
        self.init_args: dict[str, Any] = {}
        self.calls: list[str] = []

    def init_detector(self, config: str, checkpoint: str, **kwargs: Any) -> str:
        self.init_args = {"config": config, "checkpoint": checkpoint, **kwargs}
        return "model"

    def inference_detector(self, model: str, path: str) -> SimpleNamespace:
        self.calls.append(path)
        boxes, scores, labels = zip(*self.rows, strict=True) if self.rows else ((), (), ())
        return SimpleNamespace(
            pred_instances=SimpleNamespace(
                bboxes=FakeTensor(list(boxes)), scores=FakeTensor(list(scores)), labels=list(labels)
            )
        )


@pytest.fixture
def mmdet_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gt = ground_truth(images=2)
    images = tmp_path / "images"
    images.mkdir()
    (images / "000001.jpg").write_bytes(b"jpg")
    checkpoint = tmp_path / "atss.pth"
    checkpoint.write_bytes(b"weights")
    api = FakeApi([([1, 2, 3, 4], 0.3, 1), ([0, 0, 5, 5], 0.9, 0)])
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        )
    )
    modules = {"mmdet.apis": api, "torch": torch}

    def fake_import(name: str) -> Any:
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(mmdet_module, "import_module", fake_import)
    body = (
        "[detectors.atss_r50]\n"
        f'config = "{tmp_path / "atss.py"}"\n'
        f'checkpoint = "{checkpoint}"\n'
        f'checkpoint_sha256 = "{hashlib.sha256(b"weights").hexdigest()}"\n'
        "score_threshold = 0.05\nmax_per_image = 100\n"
    )
    return SimpleNamespace(
        gt=gt, images=images, api=api, torch=torch, modules=modules, body=body, tmp=tmp_path
    )


def test_mmdet_adapter_converts_pred_instances(mmdet_setup) -> None:
    s = mmdet_setup
    lab = lab_config(s.tmp, s.body)
    detector = build_detector(lab, "atss_r50", s.gt, precision="fp32_tf32_off", images_dir=s.images)
    assert s.api.init_args["cfg_options"] == {
        "model.test_cfg.score_thr": 0.05,
        "model.test_cfg.max_per_img": 100,
    }
    assert s.api.init_args["device"] == "cuda:0"
    assert s.torch.backends.cuda.matmul.allow_tf32 is False
    assert s.torch.backends.cudnn.allow_tf32 is False
    assert detector.runtime == {
        "precision": "fp32_tf32_off",
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
    }
    result = detector.predict(["1"])[0]
    assert [(d.score, d.label) for d in result.detections] == [(0.9, 0), (0.3, 1)]
    assert result.detections[1].box_xyxy == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(DetectorError, match="does not exist"):
        detector.predict(["2"])  # no image file
    with pytest.raises(DetectorError, match="not in the dataset"):
        detector.predict(["7"])


def test_mmdet_adapter_leaves_tf32_alone_for_plain_fp32(mmdet_setup) -> None:
    s = mmdet_setup
    lab = lab_config(s.tmp, s.body.replace("score_threshold = 0.05\nmax_per_image = 100\n", ""))
    detector = build_detector(lab, "atss_r50", s.gt, precision="fp32", images_dir=s.images)
    assert s.torch.backends.cudnn.allow_tf32 is True
    assert s.api.init_args["cfg_options"] is None
    assert detector.runtime["allow_tf32_cudnn"] is True  # recorded, so the run says so


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda b: b.replace('checkpoint_sha256 = "', 'checkpoint_sha256 = "0'), "expected"),
        (lambda b: b.replace("atss.pth", "missing.pth"), "does not exist"),
        (
            lambda b: b.replace("score_threshold = 0.05", 'score_threshold = "x"'),
            "must be a number",
        ),
    ],
)
def test_mmdet_adapter_refusals(mmdet_setup, edit, message: str) -> None:
    s = mmdet_setup
    lab = lab_config(s.tmp, edit(s.body))
    with pytest.raises(DetectorError, match=message):
        build_detector(lab, "atss_r50", s.gt, precision="fp32", images_dir=s.images)


def test_mmdet_adapter_needs_images_and_the_library(mmdet_setup) -> None:
    s = mmdet_setup
    lab = lab_config(s.tmp, s.body)
    with pytest.raises(DetectorError, match="needs an images directory"):
        build_detector(lab, "atss_r50", s.gt, precision="fp32")
    del s.modules["mmdet.apis"]
    with pytest.raises(DetectorError, match=r"mmdet\.apis is not importable"):
        build_detector(lab, "atss_r50", s.gt, precision="fp32", images_dir=s.images)
    del s.modules["torch"]
    with pytest.raises(DetectorError, match="torch is not importable"):
        build_detector(lab, "atss_r50", s.gt, precision="fp32", images_dir=s.images)


def test_mmdet_adapter_warns_without_a_checkpoint_hash(
    mmdet_setup, caplog: pytest.LogCaptureFixture
) -> None:
    s = mmdet_setup
    body = "\n".join(line for line in s.body.splitlines() if "sha256" not in line)
    build_detector(lab_config(s.tmp, body), "atss_r50", s.gt, precision="fp32", images_dir=s.images)
    assert "checkpoint is not verified" in caplog.text


def test_mmdet_adapter_refuses_ragged_outputs(mmdet_setup) -> None:
    s = mmdet_setup
    detector = build_detector(
        lab_config(s.tmp, s.body), "atss_r50", s.gt, precision="fp32", images_dir=s.images
    )
    s.api.inference_detector = lambda _m, _p: SimpleNamespace(  # type: ignore[method-assign]
        pred_instances=SimpleNamespace(bboxes=[[0, 0, 1, 1]], scores=[], labels=[])
    )
    with pytest.raises(DetectorError, match="differ in length"):
        detector.predict(["1"])


def test_fingerprint_follows_file_contents_and_package_versions(tmp_path: Path) -> None:
    from qcal_lab.models import detector_fingerprint

    weights = tmp_path / "w.pth"
    weights.write_bytes(b"v1")
    lab = lab_config(
        tmp_path,
        f'[detectors.atss_r50]\nconfig = "{tmp_path / "absent.py"}"\ncheckpoint = "{weights}"\n',
    )

    def fingerprint() -> dict[str, Any]:
        return detector_fingerprint(
            lab,
            "atss_r50",
            "fp32",
            file_settings=["config", "checkpoint"],
            packages=["pytest", "no-such-pkg"],
        )

    first = fingerprint()
    assert first["files"] == {"config": "missing", "checkpoint": hashlib.sha256(b"v1").hexdigest()}
    assert first["packages"]["no-such-pkg"] == "absent"
    assert first["packages"]["pytest"] == pytest.__version__
    weights.write_bytes(b"v2")  # replaced at the same path
    assert fingerprint()["files"]["checkpoint"] != first["files"]["checkpoint"]


# --- deployment targets ----------------------------------------------------------------------


def test_every_registered_kind_declares_its_targets() -> None:
    """A kind without targets could run no cell; a target for no kind is a typo."""
    assert set(KIND_TARGETS) == set(DETECTORS.names())
    assert all(KIND_TARGETS.values())


def test_detector_targets_follow_the_configured_kind(tmp_path: Path) -> None:
    lab = lab_config(tmp_path, '[detectors.mystery]\nkind = "onnx"\n')
    assert detector_targets(lab, "fixture") == {"torch_fp32"}
    assert detector_targets(lab, "mystery") == frozenset()  # an unknown kind produces nothing
    with pytest.raises(ConfigError, match="not configured"):
        detector_targets(lab, "absent")
