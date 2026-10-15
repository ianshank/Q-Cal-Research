"""Detector sanity check (cycle plan PR-F, D2/F3): the detector's raw AP against the model
zoo's published value, pass/fail, declared before it runs, never used to choose a checkpoint.

This is the scaffold. The zoo value, its tolerance and its source are Ian's (D2): empty keys
in ``[sanity.<detector>]`` of ``tooling.toml`` until he sets them. They change no run, and
the commit that sets them is the declaration. Measuring AP stays disabled until D8 decides
who may own metric code (AGENTS.md: only Ian's loop reports metrics), so a declared check is
NOT ASSESSED until then; nothing here computes or prints an AP.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from qcal.log import get_logger
from qcal.reports import FAIL, PASS
from qcal_lab.config import load_tooling

_log = get_logger("lab.sanity")

#: D8 (docs/changes/cycle-2026-10-g1-readiness.md): until it is decided, no AP is measured.
AP_MEASUREMENT_ENABLED: Final = False
NOT_ASSESSED: Final = "NOT ASSESSED"
SECTION: Final = "sanity"
DECISION: Final = "D2"


class SanityError(ValueError):
    """A sanity setting that is not a number in range."""


def _optional_number(value: Any, key: str, *, low: float, high: float) -> float | None:
    if isinstance(value, str) and not value.strip():
        return None  # not set yet
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SanityError(f"{key} must be a number or empty, got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and low <= number <= high):
        raise SanityError(f"{key} must be in [{low}, {high}], got {value!r}")
    return number


@dataclass(frozen=True)
class Expectation:
    """What Ian declared for one detector; ``None`` where he has not yet."""

    detector: str
    zoo_ap: float | None = None  # pycocotools units: 39.4 AP is 0.394
    tolerance: float | None = None
    source: str = ""

    def missing(self) -> list[str]:
        where = f"{SECTION}.{self.detector}"
        return [
            f"{where}.{key} is not set (Ian, {DECISION})"
            for key, value in (
                ("zoo_ap", self.zoo_ap),
                ("tolerance", self.tolerance),
                ("source", self.source or None),
            )
            if value is None
        ]


def expectation(detector: str, tooling: Mapping[str, Any] | None = None) -> Expectation:
    """The declared expectation for ``detector`` (``[sanity.<detector>]``)."""
    sections = (tooling if tooling is not None else load_tooling()).get(SECTION, {})
    raw = sections.get(detector, {}) if isinstance(sections, Mapping) else {}
    if not isinstance(raw, Mapping):
        raise SanityError(f"{SECTION}.{detector} must be a table")
    where = f"{SECTION}.{detector}"
    return Expectation(
        detector=detector,
        zoo_ap=_optional_number(raw.get("zoo_ap", ""), f"{where}.zoo_ap", low=0.0, high=1.0),
        tolerance=_optional_number(
            raw.get("tolerance", ""), f"{where}.tolerance", low=0.0, high=1.0
        ),
        source=str(raw.get("source", "")).strip(),
    )


@dataclass
class SanityReport:
    detector: str
    verdict: str
    reasons: list[str] = field(default_factory=list)

    @property
    def declared(self) -> bool:
        """Whether Ian declared the expectation; the precondition for a registered run."""
        return not any(r.endswith(f"(Ian, {DECISION})") for r in self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {"detector": self.detector, "verdict": self.verdict, "reasons": self.reasons}

    def render_text(self) -> str:
        return "; ".join([self.verdict, *self.reasons])


def judge(expected: Expectation, measured: float | None) -> SanityReport:
    """PASS when ``measured`` is within the declared tolerance of the zoo value."""
    zoo_ap, tolerance, missing = expected.zoo_ap, expected.tolerance, expected.missing()
    if missing or zoo_ap is None or tolerance is None:
        return SanityReport(expected.detector, NOT_ASSESSED, missing)
    if measured is None:
        return SanityReport(
            expected.detector, NOT_ASSESSED, ["AP measurement is disabled until D8"]
        )
    if not (math.isfinite(measured) and 0.0 <= measured <= 1.0):
        return SanityReport(expected.detector, FAIL, ["the measured AP is not in [0, 1]"])
    if abs(measured - zoo_ap) <= tolerance:
        return SanityReport(expected.detector, PASS)
    return SanityReport(
        expected.detector,
        FAIL,
        [f"further than {tolerance} from the declared zoo AP ({expected.source})"],
    )


def measure_ap(detector: str) -> float | None:
    """The detector's raw AP; ``None`` while D8 keeps the measurement disabled."""
    if not AP_MEASUREMENT_ENABLED:
        _log.debug("sanity check for %s: AP measurement disabled until D8", detector)
        return None
    raise NotImplementedError("the pycocotools measurement lands with D8 (PR-F)")


def check_detector(detector: str, tooling: Mapping[str, Any] | None = None) -> SanityReport:
    report = judge(expectation(detector, tooling), measure_ap(detector))
    _log.info("detector sanity check %s: %s", detector, report.verdict)
    return report


__all__ = [
    "AP_MEASUREMENT_ENABLED",
    "NOT_ASSESSED",
    "Expectation",
    "SanityError",
    "SanityReport",
    "check_detector",
    "expectation",
    "judge",
    "measure_ap",
]
