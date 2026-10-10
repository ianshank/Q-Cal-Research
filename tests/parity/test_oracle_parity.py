"""Parity with the fiveai/detection_calibration oracle on identical inputs (plan §5, Phase 1).

Cases live in ``parity.fixtures_dir`` (tests/parity/fixtures/); see tests/parity/README.md
for how they are produced. Until the oracle outputs exist, each test skips and says why:
a skipped parity test is reported as not assessed, never as passed.
"""

from __future__ import annotations

import pytest

from qcal_lab.config import load_lab_config
from qcal_lab.evaluation import HandwrittenMissingError, load_eval_loop
from qcal_lab.parity import OracleCase, check_calibrator_case, check_metric_case, load_cases
from tests.conftest import REPO_ROOT

LAB = load_lab_config(REPO_ROOT)
CASES = load_cases(REPO_ROOT / LAB.text("parity.fixtures_dir"))


def _cases(kind: str) -> list[OracleCase | None]:
    found: list[OracleCase | None] = [c for c in CASES if c.kind == kind]
    return found or [None]


def _id(case: OracleCase | None) -> str:
    return case.name if case else "no-oracle-outputs-yet"


@pytest.mark.parametrize("case", _cases("calibrator"), ids=_id)
def test_calibrator_matches_the_oracle(case: OracleCase | None) -> None:
    if case is None:
        pytest.skip("no oracle calibrator outputs yet (tests/parity/README.md)")
    assert check_calibrator_case(case, LAB) == []


@pytest.mark.parametrize("case", _cases("metric"), ids=_id)
def test_handwritten_metric_matches_the_oracle(case: OracleCase | None) -> None:
    if case is None:
        pytest.skip("no oracle metric outputs yet (tests/parity/README.md)")
    try:
        loop = load_eval_loop(LAB)
    except HandwrittenMissingError:
        pytest.skip("Ian's evaluation loop is not written yet (CLAUDE.md rule 4)")
    tolerance = LAB.config.float_value("parity.abs_tolerance")
    # A mismatch is reported to Ian; his file is never edited to make it pass.
    assert check_metric_case(case, loop, tolerance) == []
