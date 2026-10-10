"""The oracle-parity harness, the readiness report and the smoke report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qcal.config import load_config
from qcal_lab.calib import build_calibrator
from qcal_lab.config import load_lab_config
from qcal_lab.fixture_eval import FixtureEvalLoop
from qcal_lab.parity import (
    ParityError,
    check_calibrator_case,
    check_metric_case,
    compare,
    load_case,
    load_cases,
)
from qcal_lab.predictions import PredictionsHeader, write_predictions
from qcal_lab.smoke import SmokeReport, run_smoke
from qcal_lab.status import build_status
from tests.lab_support import coco_doc, det, image, lab_config

ORACLE = {"repository": "fiveai/detection_calibration", "commit": "abc123"}


def _case(directory: Path, name: str, data: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


def test_compare() -> None:
    assert compare([0.1, 0.2], [0.1, 0.2 + 1e-9], 1e-6) == []
    assert compare([0.1], [0.1, 0.2], 1e-6) == ["1 values, oracle has 2"]
    assert compare([float("nan")], [0.0], 1.0)[0].startswith("[0] nan")
    assert len(compare([1.0] * 20, [0.0] * 20, 0.1)) == 10


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"kind": "table"}, "'kind' must be one of"),
        ({"kind": "metric"}, "'oracle' must record"),
        ({"kind": "metric", "oracle": {"repository": "x", "commit": ""}}, "'oracle' must record"),
    ],
)
def test_case_validation(tmp_path: Path, data: dict[str, Any], message: str) -> None:
    with pytest.raises(ParityError, match=message):
        load_case(_case(tmp_path, "c", data))


def test_unreadable_case_and_missing_directory(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ParityError, match="cannot read"):
        load_case(tmp_path / "bad.json")
    assert load_cases(tmp_path / "none") == []


def test_calibrator_cases(tmp_path: Path) -> None:
    lab = lab_config(tmp_path)
    fit = {"scores": [0.2, 0.4, 0.6, 0.8], "targets": [0.1, 0.5, 0.3, 0.9]}
    model = build_calibrator("isotonic", lab)
    model.fit(fit["scores"], fit["targets"])
    expected = list(model.transform([0.3, 0.7]))
    good = {"kind": "calibrator", "calibrator": "isotonic", "oracle": ORACLE, "fit": fit}
    good["transform"] = {"scores": [0.3, 0.7], "expected": expected}
    case = load_case(_case(tmp_path / "f", "iso", good))
    assert case.name == "iso"
    assert check_calibrator_case(case, lab) == []
    wrong = {**good, "transform": {"scores": [0.3, 0.7], "expected": [0.0, 0.0]}, "tolerance": 0}
    assert len(check_calibrator_case(load_case(_case(tmp_path / "f", "wrong", wrong)), lab)) == 2
    for broken, message in [
        ({**good, "fit": None}, "needs 'fit' and 'transform'"),
        ({**good, "calibrator": 3}, "needs a 'calibrator'"),
        ({**good, "fit": {"scores": "x", "targets": []}}, "list of numbers"),
        ({**good, "tolerance": -1}, "tolerance"),
    ]:
        with pytest.raises(ParityError, match=message):
            check_calibrator_case(load_case(_case(tmp_path / "f", "broken", broken)), lab)
    assert [c.name for c in load_cases(tmp_path / "f")] == ["broken", "iso", "wrong"]


def test_metric_cases(tmp_path: Path) -> None:
    from qcal_lab.data.fixture import FIXTURE_DESCRIPTION

    directory = tmp_path / "cases"
    doc = coco_doc(images=1, description=FIXTURE_DESCRIPTION, boxes={1: [(1, [0, 0, 10, 10])]})
    _case(directory, "gt", doc)
    header = PredictionsHeader("x", "test", "raw", "d", "s", (1, 3))
    write_predictions(directory / "p.jsonl", header, [image("1", det(0.5, 0, (0, 0, 10, 10)))])
    case = {"kind": "metric", "oracle": ORACLE, "predictions": "p.jsonl", "ground_truth": "gt.json"}
    loop = FixtureEvalLoop()
    good = load_case(
        _case(directory, "m", {**case, "metric": "smoke_mean_abs_gap", "expected": 0.5})
    )
    assert check_metric_case(good, loop, 1e-9) == []
    absent = load_case(_case(directory, "a", {**case, "metric": "LaECE0", "expected": 1.0}))
    assert "reports no 'LaECE0'" in check_metric_case(absent, loop, 1e-9)[0]
    for broken, message in [
        ({**case, "metric": "x", "expected": True}, "numeric 'expected'"),
        ({**case, "metric": "x", "expected": 1.0, "predictions": 3}, "'predictions' path"),
    ]:
        with pytest.raises(ParityError, match=message):
            check_metric_case(load_case(_case(directory, "b", broken)), loop, 1e-9)


def test_status_of_an_empty_repository(repo: Path) -> None:
    report = build_status(load_config(repo, environ={}), load_lab_config(repo))
    assert not report.passed
    ready = {name: ok for name, ok, _ in report.items}
    assert ready["lab configuration"] is False
    assert ready["executor.command"] is False
    assert "phase 1 status: not ready" in report.render_text()
    assert report.to_dict()["ready"] is False


def test_status_reports_broken_loops_and_cases(repo: Path) -> None:
    (repo / "configs").mkdir()
    (repo / "configs/lab.toml").write_text('[eval_loop]\nmodule = "json"\nfactory = "build"\n')
    _case(repo / "tests/parity/fixtures", "x", {"kind": "nope"})
    report = build_status(load_config(repo, environ={}), load_lab_config(repo))
    details = {name: detail for name, _, detail in report.items}
    assert details["evaluation loop (Ian)"].startswith("cannot load")
    assert "'kind' must be one of" in details["oracle parity cases"]


def test_smoke_report_rendering() -> None:
    report = SmokeReport(project="p")
    assert not report.passed  # no checks is not a pass
    assert report.add("a", ok=True)
    assert not report.add("b", ok=False, detail="why")
    assert report.to_dict()["checks"][1] == {"name": "b", "passed": False, "detail": "why"}
    assert "FAIL b: why" in report.render_text()


def test_smoke_detects_a_failing_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lab = lab_config(tmp_path, '[smoke]\ncalibrators = ["none"]\n')
    report = run_smoke(lab, tmp_path / "w", python=str(tmp_path / "no-such-python"))
    assert not report.passed
    failed = {name for name, ok, _ in report.checks if not ok}
    assert "every pre-registered run succeeded" in failed


def test_smoke_refuses_incomplete_split_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qcal.config import ConfigError
    from qcal_lab import smoke

    def with_extra_split(root: Path, **_: Any) -> Any:
        return load_config(root, repo_text='[data]\nsplits = ["val", "holdout"]\n')

    monkeypatch.setattr(smoke, "load_config", with_extra_split)
    with pytest.raises(ConfigError, match=r"no size for \['holdout'\]"):
        run_smoke(lab_config(tmp_path), tmp_path / "w")


def test_smoke_refuses_split_sizes_larger_than_the_fixture(tmp_path: Path) -> None:
    from qcal_lab.data.splits import SplitError

    with pytest.raises(SplitError, match="only 20 exist"):
        run_smoke(lab_config(tmp_path, "[smoke.split_sizes]\ntest = 100\n"), tmp_path / "w")
