"""Every check report renders and exits through one protocol (qcal.reports.CheckReport)."""

from __future__ import annotations

import pytest

from qcal.ci.immutability import ImmutabilityReport
from qcal.ci.review import ReviewReport
from qcal.ci.signatures import SignatureReport
from qcal.integrity.agent_layer import AgentLayerReport
from qcal.integrity.leakage import LeakageReport
from qcal.integrity.licenses import LicenseReport
from qcal.reports import FAIL, PASS, CheckReport, verdict
from qcal_lab.smoke import SmokeReport

REPORTS = [
    ImmutabilityReport,
    ReviewReport,
    SignatureReport,
    AgentLayerReport,
    LeakageReport,
    LicenseReport,
    SmokeReport,
]


def test_verdict_labels() -> None:
    assert (verdict(True), verdict(False)) == (PASS, FAIL)


@pytest.mark.parametrize("cls", REPORTS, ids=lambda c: c.__name__)
def test_report_satisfies_the_protocol(cls: type) -> None:
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(cls)}
    required = {
        name: "x"
        for name, f in fields.items()
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    report = cls(**required)
    assert isinstance(report, CheckReport)
    payload = report.to_dict()
    assert payload["verdict"] in {PASS, FAIL, "SKIP"}
    assert isinstance(report.render_text(), str)
