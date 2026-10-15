"""Synthetic protocol cases (PR-F, wave 3): each case really separates what it claims to.

The geometry is checked with exact fractions; no metric value is computed or compared.
"""

from __future__ import annotations

import io
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from qcal.config import load_config
from qcal.policy import Policy
from qcal_lab import cli, protocol_cases
from qcal_lab.config import load_lab_config
from qcal_lab.data.coco import parse_coco
from qcal_lab.protocol_cases import CaseSettings, build_cases, drift, iou_xywh, write_cases
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.rule("C1")


@pytest.fixture(scope="module")
def settings() -> CaseSettings:
    return CaseSettings.load(load_lab_config(REPO_ROOT))


@pytest.fixture(scope="module")
def cases(settings: CaseSettings) -> dict[str, dict[str, Any]]:
    return {
        name.removesuffix(".json"): json.loads(data) for name, data in build_cases(settings).items()
    }


def _boxes(case: dict[str, Any]) -> list[list[int]]:
    return [a["bbox"] for a in case["ground_truth"]["annotations"]]


def _dets(case: dict[str, Any]) -> list[dict[str, Any]]:
    return case["detections"]


# --- the committed files ----------------------------------------------------------------------


def test_the_committed_cases_match_the_generator(settings: CaseSettings) -> None:
    assert drift(REPO_ROOT / settings.directory, build_cases(settings)) == []


def test_the_cases_are_not_ians_data(settings: CaseSettings) -> None:
    policy = Policy.from_config(load_config(REPO_ROOT, environ={}))
    probe = f"{settings.directory}/{protocol_cases.BUILDERS[0].__name__}.json"
    assert "ian_data" not in policy.categories_for(probe)


def test_drift_names_missing_extra_and_changed_files(
    tmp_path: Path, settings: CaseSettings
) -> None:
    files = build_cases(settings)
    assert drift(tmp_path / "absent", files) == [f"{n}: missing" for n in sorted(files)]
    write_cases(tmp_path, files)
    assert drift(tmp_path, files) == []
    first = min(files)
    (tmp_path / first).write_bytes(files[first] + b" ")
    (tmp_path / "stray.json").write_text("{}\n", "utf-8")
    (tmp_path / "notes.md").write_text("not a case\n", "utf-8")
    assert drift(tmp_path, files) == [
        "stray.json: not generated",
        f"{first}: differs from the generator",
    ]
    assert write_cases(tmp_path, files) == [first]  # only what changed is rewritten


def test_the_command_writes_and_checks(tmp_path: Path) -> None:
    def lab(*args: str) -> tuple[int, str]:
        out = io.StringIO()
        return cli.main(
            ["--root", str(REPO_ROOT), "protocol-cases", *args], out=out
        ), out.getvalue()

    target = tmp_path / "cases"
    assert lab("--out", str(target), "--check")[0] == cli.EXIT_FAILED
    code, text = lab("--out", str(target))
    assert (code, "7 written" in text) == (cli.EXIT_OK, True)
    assert lab("--out", str(target), "--check") == (cli.EXIT_OK, "protocol cases: PASS\n")
    assert lab("--check")[0] == cli.EXIT_OK  # the committed cases


def test_bad_settings_are_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    lab = load_lab_config(REPO_ROOT)
    for override, message in [
        ({"iou_threshold": 1.0}, "iou_threshold must be in (0, 1)"),
        ({"iou_threshold": 0.99}, "too close to 1"),
        ({"bin_count": 1}, "bin_count >= 2"),
        ({"image_width": 50}, "too small"),
    ]:
        with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
            CaseSettings.load(lab, override)
    stops_at_one = CaseSettings.load(lab).__class__(
        **{**vars(CaseSettings.load(lab)), "grid": (0.0, 1.0)}
    )
    assert stops_at_one.problems() == [
        "the threshold grid must stop below 1 (no score lies above it)"
    ]
    original = protocol_cases.load_tooling
    monkeypatch.setattr(
        protocol_cases,
        "load_tooling",
        lambda: {**original(), "protocol_cases": {**original()["protocol_cases"], "bin_count": 0}},
    )
    out = io.StringIO()
    assert cli.main(["--root", str(REPO_ROOT), "protocol-cases"], out=out) == cli.EXIT_USAGE


# --- every case ---------------------------------------------------------------------------------


def test_every_case_is_valid_coco_with_scored_detections(cases: dict[str, dict[str, Any]]) -> None:
    for name, case in cases.items():
        ground_truth = parse_coco(case["ground_truth"])
        categories = set(ground_truth.category_ids)
        for det in _dets(case):
            assert set(det) == {"image_id", "category_id", "bbox", "score"}, name
            assert det["category_id"] in categories, name
            assert 0.0 < det["score"] <= 1.0, name


def test_every_case_separates_its_alternatives(cases: dict[str, dict[str, Any]]) -> None:
    assert len(cases) == len(protocol_cases.BUILDERS) == 7
    for name, case in cases.items():
        predictions = [json.dumps(a["predicts"], sort_keys=True) for a in case["alternatives"]]
        assert len(predictions) >= 2, name
        assert len(set(predictions)) == len(predictions), name
        for alternative in case["alternatives"]:
            assert set(alternative["predicts"]) <= protocol_cases.PREDICATES, name
            for detection in alternative["predicts"].get("outcomes", {}):
                assert 0 <= int(detection) < len(_dets(case)), name


# --- each case's claim, exactly ---------------------------------------------------------------


def test_iou_is_exactly_the_threshold(
    cases: dict[str, dict[str, Any]], settings: CaseSettings
) -> None:
    case = cases["iou_at_threshold"]
    (box,), (det,) = _boxes(case), _dets(case)
    iou = iou_xywh(det["bbox"], box)
    assert iou == Fraction(repr(settings.iou_threshold))
    # In floats too: what the oracle computes (pycocotools divides intersection by union).
    inter = det["bbox"][2] * det["bbox"][3]
    assert inter / (box[2] * box[3]) == settings.iou_threshold


@pytest.mark.parametrize("threshold", [0.5, 0.55, 0.7, 0.75])
def test_the_iou_case_is_exact_for_other_thresholds(threshold: float) -> None:
    roomy = {"iou_threshold": threshold, "image_width": 400, "image_height": 400}
    settings = CaseSettings.load(load_lab_config(REPO_ROOT), roomy)
    case = protocol_cases.iou_at_threshold(settings)
    assert iou_xywh(case.detections[0]["bbox"], case.annotations[0]["bbox"]) == Fraction(
        repr(threshold)
    )
    first = protocol_cases.equal_score_order(settings)
    assert (
        Fraction(threshold)
        < iou_xywh(first.detections[0]["bbox"], first.annotations[0]["bbox"])
        < 1
    )


def test_equal_scores_compete_and_the_worse_overlap_comes_first(
    cases: dict[str, dict[str, Any]], settings: CaseSettings
) -> None:
    case = cases["equal_score_order"]
    (box,), (first, second) = _boxes(case), _dets(case)
    assert first["score"] == second["score"]
    worse, better = iou_xywh(first["bbox"], box), iou_xywh(second["bbox"], box)
    assert Fraction(settings.iou_threshold) < worse < better == 1


def test_the_detection_overlaps_only_a_crowd_region(cases: dict[str, dict[str, Any]]) -> None:
    case = cases["crowd_only_overlap"]
    (annotation,), (det,) = case["ground_truth"]["annotations"], _dets(case)
    assert annotation["iscrowd"] == 1
    x, y, w, h = det["bbox"]
    ax, ay, aw, ah = annotation["bbox"]
    assert ax <= x  # fully inside
    assert ay <= y
    assert x + w <= ax + aw
    assert y + h <= ay + ah


def test_scores_lie_exactly_on_bin_edges(
    cases: dict[str, dict[str, Any]], settings: CaseSettings
) -> None:
    case = cases["scores_on_bin_edges"]
    bins = settings.bin_count
    for det in _dets(case):
        k = round(det["score"] * bins)
        assert det["score"] == k / bins
        assert 1 <= k <= bins
    assert max(d["score"] for d in _dets(case)) == 1.0
    left, right = (a["predicts"]["bins"] for a in case["alternatives"])
    assert all(left[i] != right[i] for i in left if _dets(case)[int(i)]["score"] < 1.0)


def test_one_class_has_objects_and_no_detections(cases: dict[str, dict[str, Any]]) -> None:
    case = cases["class_without_detections"]
    with_objects = {a["category_id"] for a in case["ground_truth"]["annotations"]}
    detected = {d["category_id"] for d in _dets(case)}
    assert with_objects - detected == {2}


def test_the_tie_straddles_the_cut(
    cases: dict[str, dict[str, Any]], settings: CaseSettings
) -> None:
    dets = _dets(cases["ties_at_max_detections"])
    cut = settings.max_detections
    scores = [d["score"] for d in dets]
    assert len(dets) == cut + 1
    assert scores[cut - 1] == scores[cut]  # the last kept and the first cut tie
    assert all(a > b for a, b in itertools.pairwise(scores[:cut]))
    boxes = [tuple(d["bbox"]) for d in dets]
    assert len(set(boxes)) == len(boxes)  # distinct boxes, so nothing else ties
    assert all(iou_xywh(list(a), list(b)) == 0 for i, a in enumerate(boxes) for b in boxes[i + 1 :])


def test_every_grid_threshold_keeps_the_same_detections(
    cases: dict[str, dict[str, Any]], settings: CaseSettings
) -> None:
    case = cases["threshold_selection_ties"]
    kept = {t: [d["score"] >= t for d in _dets(case)] for t in settings.grid}
    assert all(all(k) for k in kept.values())  # every threshold keeps every detection
    lowest, highest = (a["predicts"]["threshold"] for a in case["alternatives"])
    assert (lowest, highest) == (settings.grid[0], settings.grid[-1])
