"""Golden digests for every identity a run record or cache entry depends on (PR-D1).

Each value was computed at `8c76044`, before any run-identity change. A failing test here
means a run's identity changed. That is allowed only when a change proposal in
`docs/changes/` says which digest changes and why; re-pin the value in the same commit and
name the proposal next to it.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from qcal.config import load_config
from qcal.protocols import Detection, ImageDetections
from qcal.registry.cells import cell_id
from qcal.registry.experiments import load_experiments
from qcal.registry.runner import Runner
from qcal.registry.store import RegistryStore
from qcal_lab.config import load_lab_config
from qcal_lab.data.coco import load_coco
from qcal_lab.data.fixture import build_fixture, fixture_bytes, write_fixture
from qcal_lab.data.splits import draw, rank_key, split_digest
from qcal_lab.experiment import _effective_digest
from qcal_lab.models import build_detector
from qcal_lab.predictions import PredictionsHeader, cache_key, dumps
from tests.conftest import FakeExecutor

pytestmark = pytest.mark.rule("C7")

REPO_ROOT = Path(__file__).resolve().parents[2]
IDS = [f"img{i:03d}" for i in range(50)]
FIXTURE = {
    "images": 4,
    "categories": 3,
    "max_objects_per_image": 3,
    "seed": 11,
    "width": 320,
    "height": 240,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_cell_id() -> None:
    factors = {"detector": "atss_r50", "calibrator": "platt", "seed_note": 1}
    assert cell_id(factors, prefix="C-", length=12) == "C-ef66196eb152"


def test_split_digest_rank_key_and_draw() -> None:
    assert split_digest(IDS) == "02ab27ee7c966d48e4427c3a1b1e094b521d2cb8e92d9054ff8cc784878cf622"
    assert rank_key(7, "img001") == (
        "d099b68bad4df68a06a647606f46d6c58cb9fd032cb64e7b7dea70f342ae7992"
    )
    assert draw(IDS, 5, 7) == ("img002", "img031", "img032", "img044", "img047")


def test_config_hash(tmp_path: Path) -> None:
    (tmp_path / "qcal.toml").write_text(
        '[executor]\ncommand = ["prog", "--run", "{run_id}", "--seed", "{seed}"]\n',
        encoding="utf-8",
    )
    (tmp_path / "EXPERIMENTS.yaml").write_text(
        textwrap.dedent(
            """
            version: 2
            seeds: [0, 1]
            cells:
              - id: C-golden
                detector: atss_r50
                calibrator: platt
            """
        ),
        encoding="utf-8",
    )
    config = load_config(tmp_path, environ={})
    experiments = load_experiments(config)
    store = RegistryStore(config.path("registry_dir"))
    runner = Runner(config, experiments, store, FakeExecutor(), collectors=[])
    digest = runner.config_hash(
        experiments.cell("C-golden"), 1, inputs={"configs/lab.toml": "ab" * 32}
    )
    assert digest == "3b2cd7139c2cf90aa7252df5d4a9f59a8f17a98a5da86eeeef7a07c447e4748c"


def test_fixture_bytes() -> None:
    document = build_fixture(**FIXTURE)
    assert sha(fixture_bytes(document)) == (
        "dd16e6ba0fd75dcac7a4a757613cddcf4615f4b17459dfd5ea57c1540ec316a9"
    )


def test_predictions_header_and_rows() -> None:
    header = PredictionsHeader(
        detector="golden",
        split="val",
        stage="raw",
        dataset_sha256="d" * 64,
        split_sha256="e" * 64,
        category_ids=(1, 2, 3),
    )
    images = [
        ImageDetections("2", (Detection((1.0, 2.0, 3.5, 4.25), 0.75, 1, None),)),
        ImageDetections(
            "1",
            (
                Detection((0.0, 0.0, 10.0, 10.0), 0.5, 0, -0.25),
                Detection((5.0, 5.0, 6.0, 7.0), 1.0, 2, None),
            ),
        ),
        ImageDetections("3", ()),
    ]
    assert sha(dumps(header, images)) == (
        "59086c04a9cc71a2a92fda31abad82aebfec6a2832bee5c5be74461a484d345f"
    )


def test_cache_key() -> None:
    key = cache_key(
        detector={"name": "golden", "precision": "fp32"},
        dataset_sha256="d" * 64,
        split_sha256="e" * 64,
        images_dir="",
        images_sha256="",
        source_sha256="f" * 64,
    )
    assert key == "6c009b641570586760fdb74a30d0bf01a0877a6a8926fe52e8d2532c79811b8c"


def test_lab_config_effective_digest() -> None:
    lab = load_lab_config(REPO_ROOT)
    assert _effective_digest(lab) == (
        "d10350f08925addc61157b1ba957eb0ed24cf3d530dbec1eafaa2ba7494a40aa"
    )


def test_fixture_detector_predictions(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    write_fixture(path, build_fixture(**FIXTURE))
    ground_truth = load_coco(path)
    lab = load_lab_config(REPO_ROOT)
    detector = build_detector(lab, "fixture", ground_truth, precision="fp32")
    predictions = tuple(detector.predict(sorted(ground_truth.images)))
    header = PredictionsHeader(
        "fixture",
        "val",
        "raw",
        ground_truth.sha256,
        split_digest(ground_truth.images),
        tuple(ground_truth.category_ids),
    )
    assert sha(dumps(header, predictions)) == (
        "f4184d55f40bb462dcac430effa4ca6903f8b0d262eae02c6d9a18f930af301b"
    )
