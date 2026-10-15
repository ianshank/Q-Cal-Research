"""Detector sanity-check scaffold (PR-F, wave 3): pass/fail plumbing; Ian's values are empty.

The values used below are made up to drive the verdicts; they are not zoo values.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from qcal.config import load_config
from qcal_lab import sanity, status
from qcal_lab.config import load_lab_config, load_tooling
from qcal_lab.sanity import NOT_ASSESSED, Expectation, SanityError, expectation, judge
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.rule("C1")

DECLARED = Expectation("det", zoo_ap=0.5, tolerance=0.01, source="a model card")


def _tooling(**values: Any) -> dict[str, Any]:
    return {"sanity": {"det": values}}


def test_the_packaged_values_are_left_for_ian() -> None:
    declared = expectation("atss_r50")
    assert (declared.zoo_ap, declared.tolerance, declared.source) == (None, None, "")
    assert load_tooling()["sanity"]["atss_r50"] == {"zoo_ap": "", "tolerance": "", "source": ""}
    assert sanity.AP_MEASUREMENT_ENABLED is False  # D8


def test_an_undeclared_check_is_not_assessed_and_names_what_is_missing() -> None:
    report = judge(Expectation("det", zoo_ap=0.5), measured=0.5)
    assert report.verdict == NOT_ASSESSED
    assert report.reasons == [
        "sanity.det.tolerance is not set (Ian, D2)",
        "sanity.det.source is not set (Ian, D2)",
    ]
    assert not report.declared


def test_a_declared_check_is_not_assessed_until_d8() -> None:
    report = sanity.check_detector("det", _tooling(zoo_ap=0.5, tolerance=0.01, source="card"))
    assert (report.verdict, report.reasons) == (
        NOT_ASSESSED,
        ["AP measurement is disabled until D8"],
    )
    assert report.declared


EDGE = 2.0**-20  # binary values, so the boundary is exact


@pytest.mark.parametrize(
    ("measured", "verdict"),
    [(0.5, "PASS"), (0.75, "PASS"), (0.25, "PASS"), (0.75 + EDGE, "FAIL"), (0.25 - EDGE, "FAIL")],
)
def test_the_tolerance_is_inclusive(measured: float, verdict: str) -> None:
    exact = Expectation("det", zoo_ap=0.5, tolerance=0.25, source="card")
    assert judge(exact, measured).verdict == verdict


@pytest.mark.parametrize("measured", [math.nan, math.inf, -0.1, 1.5])
def test_an_impossible_measurement_fails(measured: float) -> None:
    report = judge(DECLARED, measured)
    assert (report.verdict, report.reasons) == ("FAIL", ["the measured AP is not in [0, 1]"])


def test_a_failure_names_the_source_but_no_number_it_measured() -> None:
    report = judge(DECLARED, 0.9)
    assert report.verdict == "FAIL"
    assert report.reasons == ["further than 0.01 from the declared zoo AP (a model card)"]
    assert "0.9" not in report.render_text()
    assert report.to_dict() == {"detector": "det", "verdict": "FAIL", "reasons": report.reasons}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"zoo_ap": 39.4}, "zoo_ap must be in"),  # percent, not pycocotools units
        ({"zoo_ap": True}, "must be a number"),
        ({"tolerance": "0.1"}, "must be a number"),
        ({"tolerance": -0.1}, "tolerance must be in"),
        ({"zoo_ap": math.nan}, "zoo_ap must be in"),
    ],
)
def test_bad_values_are_refused(values: dict[str, Any], message: str) -> None:
    with pytest.raises(SanityError, match=message):
        expectation("det", _tooling(**values))


def test_a_section_that_is_not_a_table_is_refused() -> None:
    with pytest.raises(SanityError, match="must be a table"):
        expectation("det", {"sanity": {"det": "0.5"}})
    assert expectation("det", {}) == Expectation("det")


def test_the_measurement_stays_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sanity.measure_ap("det") is None
    monkeypatch.setattr(sanity, "AP_MEASUREMENT_ENABLED", True)
    with pytest.raises(NotImplementedError, match="D8"):
        sanity.measure_ap("det")


def _sanity_item(report: status.StatusReport, name: str) -> dict[str, Any]:
    (item,) = [i for i in report.to_dict()["items"] if i["name"] == f"detector sanity check {name}"]
    return item


@pytest.mark.parametrize(
    ("values", "ready"),
    [
        ({}, False),  # not declared: a registered run would let the value be chosen after
        ({"zoo_ap": 0.5, "tolerance": 0.01, "source": "card"}, True),  # declared; D8 pending
        ({"zoo_ap": "x"}, False),  # a bad value is reported, not raised
    ],
)
def test_status_needs_the_declaration(
    monkeypatch: pytest.MonkeyPatch, values: dict[str, Any], ready: bool
) -> None:
    monkeypatch.setattr(sanity, "load_tooling", lambda: {"sanity": {"atss_r50": values}})
    report = status.build_status(load_config(REPO_ROOT, environ={}), load_lab_config(REPO_ROOT))
    assert _sanity_item(report, "atss_r50")["ready"] is ready
