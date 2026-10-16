"""Detector sanity check (PR-F, wave 3; rev. 3): declared before a run, bound to what it was
measured on, compared in the declared units; Ian's values are empty.

The values used below are made up to drive the verdicts; they are not zoo values.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import pytest

from qcal.config import load_config
from qcal_lab import declarations, sanity, status
from qcal_lab.sanity import (
    NOT_ASSESSED,
    Configured,
    Expectation,
    SanityError,
    expectation,
    judge,
    require_declared,
)
from tests.conftest import REPO_ROOT
from tests.lab_support import lab_config

pytestmark = pytest.mark.rule("C1")

CHECKPOINT = "c" * 64
TEST_CFG = {"score_threshold": 0.5, "max_per_image": 7}
CONFIGURED = Configured(CHECKPOINT, TEST_CFG)


def declared(**overrides: Any) -> dict[str, Any]:
    """A complete declaration bound to ``CONFIGURED``, in fractions."""
    values: dict[str, Any] = {
        "zoo_ap": 0.7,
        "tolerance": 0.04,
        "units": "fraction",
        "source": "a model card",
        "checkpoint_sha256": CHECKPOINT,
        "image_set": "a held-out set",
        "test_cfg": dict(TEST_CFG),
        "on_fail": "stop",
    }
    values.update(overrides)
    return values


def test_nothing_is_declared_until_ian_declares_it() -> None:
    assert expectation("atss_r50", {}) == Expectation("atss_r50")
    assert expectation("atss_r50", {}).missing() == [
        f"detectors.atss_r50.{key} is not set (Ian, D2)"
        for key in (
            "zoo_ap",
            "tolerance",
            "units",
            "source",
            "checkpoint_sha256",
            "image_set",
            "test_cfg",
            "on_fail",
        )
    ]
    assert sanity.AP_MEASUREMENT_ENABLED is False  # D8


@pytest.mark.parametrize(
    "key", ["zoo_ap", "units", "source", "checkpoint_sha256", "image_set", "test_cfg", "on_fail"]
)
def test_an_incomplete_declaration_is_not_assessed_and_names_what_is_missing(key: str) -> None:
    values = declared()
    del values[key]
    report = judge(expectation("det", values), measured=0.7, configured=CONFIGURED)
    assert (report.verdict, report.declared) == (NOT_ASSESSED, False)
    assert report.reasons == [f"detectors.det.{key} is not set (Ian, D2)"]


def test_a_declared_check_is_not_assessed_until_d8() -> None:
    report = sanity.check_detector("det", declared(), CONFIGURED)
    assert (report.verdict, report.reasons, report.declared) == (
        NOT_ASSESSED,
        ["AP measurement is disabled until D8"],
        True,
    )


@pytest.mark.parametrize(
    ("units", "zoo_ap", "tolerance", "measured", "verdict"),
    [
        # 0.8 - 0.7 is 0.10000000000000009 in binary floats: compared as written, it is 0.1.
        ("fraction", 0.7, 0.1, 0.8, "PASS"),
        ("fraction", 0.7, 0.1, 0.6, "PASS"),
        ("fraction", 0.7, 0.1, 0.7, "PASS"),
        ("fraction", 0.7, 0.1, 0.8000001, "FAIL"),
        ("fraction", 0.7, 0.1, 0.5999999, "FAIL"),
        ("percent", 70, 1, 0.71, "PASS"),  # pycocotools' 0.71 is 71 percent
        ("percent", 70, 1, 0.69, "PASS"),
        ("percent", 70, 1, 0.7101, "FAIL"),
        ("percent", 70.0, 0, 0.7, "PASS"),
    ],
)
def test_the_tolerance_is_inclusive_in_the_declared_units(
    units: str, zoo_ap: float, tolerance: float, measured: float, verdict: str
) -> None:
    cap = Decimal("0.1")
    values = declared(units=units, zoo_ap=zoo_ap, tolerance=tolerance)
    assert judge(expectation("det", values, cap=cap), measured).verdict == verdict


@pytest.mark.parametrize("measured", [math.nan, math.inf, -0.1, 1.5])
def test_an_impossible_measurement_fails(measured: float) -> None:
    report = judge(expectation("det", declared()), measured)
    assert (report.verdict, report.reasons) == ("FAIL", ["the measured AP is not in [0, 1]"])


def test_a_failure_names_the_source_and_the_action_but_no_number_it_measured() -> None:
    report = judge(expectation("det", declared()), 0.9)
    assert report.verdict == "FAIL"
    assert report.reasons == [
        "further than 0.04 fraction from the declared zoo AP (a model card); on_fail: stop"
    ]
    assert "0.9" not in report.render_text()
    assert report.to_dict() == {
        "detector": "det",
        "verdict": "FAIL",
        "declared": True,
        "reasons": report.reasons,
    }


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"zoo_ap": 70.0}, r"zoo_ap must be in \[0, 1\] fraction"),  # percent in a fraction
        ({"units": "percent", "zoo_ap": 101}, r"zoo_ap must be in \[0, 100\] percent"),
        ({"zoo_ap": True}, "must be a number"),
        ({"tolerance": "0.1"}, "must be a number"),
        ({"tolerance": -0.01}, "tolerance must be in"),
        ({"tolerance": 0.06}, r"tolerance must be in \[0, 0.05\] fraction"),  # over the cap
        ({"units": "percent", "tolerance": 6}, r"tolerance must be in \[0, 5.00\] percent"),
        ({"zoo_ap": math.nan}, "must be finite"),
        ({"units": "points"}, "units must be one of"),
        ({"on_fail": "use another checkpoint"}, "never selects another checkpoint"),
        ({"test_cfg": "score_threshold=0"}, "test_cfg must be a table"),
        ({"source": 3}, "source must be a string"),
    ],
)
def test_bad_values_are_refused(values: dict[str, Any], message: str) -> None:
    with pytest.raises(SanityError, match=message):
        expectation("det", declared(**values), cap=Decimal("0.05"))


def test_a_percent_declaration_is_accepted_in_its_own_range() -> None:
    values = declared(units="percent", zoo_ap=70, tolerance=1)
    assert expectation("det", values).zoo_ap == Decimal(70)


@pytest.mark.parametrize(
    ("configured", "reason"),
    [
        (Configured("d" * 64, TEST_CFG), "the declaration is about another checkpoint"),
        (Configured("", TEST_CFG), "the declaration is about another checkpoint"),
        (Configured(CHECKPOINT, {**TEST_CFG, "max_per_image": 8}), "measured under other"),
    ],
)
def test_a_declaration_about_something_else_is_not_one(configured: Configured, reason: str) -> None:
    report = judge(expectation("det", declared()), 0.7, configured)
    assert (report.verdict, report.declared) == (NOT_ASSESSED, False)
    (text,) = report.reasons
    assert reason in text
    assert CHECKPOINT not in text  # digests are never printed
    assert "d" * 64 not in text


def test_a_run_requires_a_complete_bound_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SanityError, match="no complete sanity declaration"):
        require_declared("det", {}, CONFIGURED)
    with pytest.raises(SanityError, match="another checkpoint"):
        require_declared("det", declared(checkpoint_sha256="e" * 64), CONFIGURED)
    assert require_declared("det", declared(), CONFIGURED).declared
    monkeypatch.setattr(sanity, "measure_ap", lambda _name: 0.1)  # once D8 measures
    with pytest.raises(SanityError, match="failed its sanity check"):
        require_declared("det", declared(), CONFIGURED)


def test_the_tolerance_cap_comes_from_the_tooling_settings() -> None:
    assert sanity.max_tolerance() == Decimal("0.05")
    assert sanity.max_tolerance({"sanity": {"max_tolerance_fraction": 0.1}}) == Decimal("0.1")
    for broken in ({}, {"sanity": {"max_tolerance_fraction": 0}}, {"sanity": []}):
        with pytest.raises(SanityError, match="max_tolerance_fraction"):
            sanity.max_tolerance(broken)


def test_only_real_detectors_need_a_declaration() -> None:
    lab = lab_config(REPO_ROOT)
    assert sanity.needs_declaration(lab, "atss_r50")
    assert not sanity.needs_declaration(lab, "fixture")
    bound = lab_config(REPO_ROOT, BOUND_LAB)
    assert sanity.configured(bound, "atss_r50") == CONFIGURED


def test_the_measurement_stays_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sanity.measure_ap("det") is None
    monkeypatch.setattr(sanity, "AP_MEASUREMENT_ENABLED", True)
    with pytest.raises(NotImplementedError, match="D8"):
        sanity.measure_ap("det")


BOUND_LAB = (
    f'[detectors.atss_r50]\ncheckpoint_sha256 = "{CHECKPOINT}"\n'
    "score_threshold = 0.5\nmax_per_image = 7\n"
)


def _sanity_item(report: status.StatusReport, name: str) -> dict[str, Any]:
    (item,) = [
        i
        for i in report.to_dict()["items"]
        if i["name"] == f"detector sanity check {name} declared"
    ]
    return item


@pytest.mark.parametrize(
    ("values", "ready"),
    [
        ({}, False),  # not declared: a registered run would let the value be chosen after
        (declared(), True),  # declared and bound; D8 pending
        (declared(checkpoint_sha256="f" * 64), False),  # about another checkpoint
        ({"zoo_ap": "x"}, False),  # a bad value is reported, not raised
    ],
)
def test_status_needs_the_declaration(
    monkeypatch: pytest.MonkeyPatch, values: dict[str, Any], ready: bool
) -> None:
    monkeypatch.setattr(
        declarations, "load_declarations", lambda *_: {"detectors": {"atss_r50": values}}
    )
    lab = lab_config(REPO_ROOT, BOUND_LAB)
    report = status.build_status(load_config(REPO_ROOT, environ={}), lab)
    item = _sanity_item(report, "atss_r50")
    assert item["ready"] is ready
    if ready:
        assert item["detail"].startswith("NOT ASSESSED")  # declared, never "passed"
        # the repository's qcal.toml is still in bootstrap mode: nobody checked who signed it
        assert item["detail"].endswith("its signer is not checked)")


def test_status_reports_an_unprotected_declarations_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def unprotected(*_: Any) -> Any:
        raise declarations.DeclarationsError("notes/d.toml is not a file agents are kept out of")

    monkeypatch.setattr(declarations, "load_declarations", unprotected)
    report = status.build_status(load_config(REPO_ROOT, environ={}), lab_config(REPO_ROOT))
    item = _sanity_item(report, "atss_r50")
    assert (item["ready"], "kept out of" in item["detail"]) == (False, True)
