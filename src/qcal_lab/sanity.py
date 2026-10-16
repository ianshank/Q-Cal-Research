"""Detector sanity check (cycle plan PR-F, D2/F3): the detector's raw AP against the model
zoo's published value, pass/fail, declared before it runs, never used to choose a checkpoint.

This is the scaffold. The zoo value, its tolerance, its units and its source are Ian's (D2):
he declares them in his declarations file (:mod:`qcal_lab.declarations`), which agents cannot
write and only his signed commits change; that commit is the declaration. A declaration is
bound to what it was measured on: the checkpoint's sha256, the image set, and the test-time
settings, which must equal the detector's configured ones. It also states what a FAIL means
(``on_fail``): the only accepted action is to stop and debug, never to swap the checkpoint.

A non-fixture detector does not run until its declaration is complete and bound
(:func:`require_declared`), so the expectation cannot be chosen after the result. Measuring AP
stays disabled until D8 decides who may own metric code (AGENTS.md: only Ian's loop reports
metrics), so a declared check is NOT ASSESSED until then; nothing here computes or prints an AP.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from qcal.log import get_logger
from qcal.reports import FAIL, PASS
from qcal_lab.config import LabConfig, load_tooling
from qcal_lab.models.base import detector_settings, detector_test_cfg

_log = get_logger("lab.sanity")

#: D8 (docs/changes/cycle-2026-10-g1-readiness.md): until it is decided, no AP is measured.
AP_MEASUREMENT_ENABLED: Final = False
NOT_ASSESSED: Final = "NOT ASSESSED"
SECTION: Final = "detectors"  # in Ian's declarations file
DECISION: Final = "D2"
#: How many declared units one unit of pycocotools AP (a fraction) is.
UNITS: Final = {"fraction": Decimal(1), "percent": Decimal(100)}
#: What a FAIL obliges; the only action that cannot become checkpoint selection.
ON_FAIL: Final = ("stop",)
TOOLING_SECTION: Final = "sanity"
#: Detector kinds no zoo value exists for: the synthetic fixture only runs on the fixture.
EXEMPT_KINDS: Final = frozenset({"fixture"})


class SanityError(ValueError):
    """A sanity declaration that is malformed, incomplete or bound to something else."""


def max_tolerance(tooling: Mapping[str, Any] | None = None) -> Decimal:
    """The largest tolerance a declaration may give, as a fraction (a plausibility cap)."""
    settings = (tooling if tooling is not None else load_tooling()).get(TOOLING_SECTION, {})
    value = settings.get("max_tolerance_fraction") if isinstance(settings, Mapping) else None
    cap = _decimal(value, f"{TOOLING_SECTION}.max_tolerance_fraction")
    if cap is None or not Decimal(0) < cap <= Decimal(1):
        raise SanityError(f"{TOOLING_SECTION}.max_tolerance_fraction must be in (0, 1]")
    return cap


def _decimal(value: Any, key: str) -> Decimal | None:
    """``value`` as written (``0.1`` stays 0.1, not its binary neighbour); ``None`` if unset."""
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SanityError(f"{key} must be a number or empty, got {value!r}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:  # pragma: no cover - str() of a number always parses
        raise SanityError(f"{key} is not a number: {value!r}") from exc
    if not number.is_finite():
        raise SanityError(f"{key} must be finite, got {value!r}")
    return number


def _text(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise SanityError(f"{key} must be a string, got {value!r}")
    return value.strip()


@dataclass(frozen=True)
class Expectation:
    """What Ian declared for one detector; empty where he has not yet."""

    detector: str
    zoo_ap: Decimal | None = None  # in ``units``
    tolerance: Decimal | None = None  # in ``units``
    units: str = ""
    source: str = ""
    checkpoint_sha256: str = ""
    image_set: str = ""
    test_cfg: Mapping[str, Any] | None = None
    on_fail: str = ""

    def missing(self) -> list[str]:
        where = f"{SECTION}.{self.detector}"
        values: dict[str, object] = {
            "zoo_ap": self.zoo_ap,
            "tolerance": self.tolerance,
            "units": self.units,
            "source": self.source,
            "checkpoint_sha256": self.checkpoint_sha256,
            "image_set": self.image_set,
            "test_cfg": self.test_cfg,
            "on_fail": self.on_fail,
        }
        return [
            f"{where}.{key} is not set (Ian, {DECISION})"
            for key, value in values.items()
            if value is None or value == ""
        ]


def expectation(
    detector: str, declared: Mapping[str, Any], *, cap: Decimal | None = None
) -> Expectation:
    """The expectation Ian declared for ``detector`` (its table in his declarations file)."""
    where = f"{SECTION}.{detector}"
    units = _text(declared.get("units", ""), f"{where}.units")
    if units and units not in UNITS:
        raise SanityError(f"{where}.units must be one of {sorted(UNITS)}, got {units!r}")
    on_fail = _text(declared.get("on_fail", ""), f"{where}.on_fail")
    if on_fail and on_fail not in ON_FAIL:
        raise SanityError(
            f"{where}.on_fail must be one of {list(ON_FAIL)}: a failed check stops the work "
            f"for debugging and never selects another checkpoint, got {on_fail!r}"
        )
    test_cfg = declared.get("test_cfg")
    if test_cfg is not None and not isinstance(test_cfg, Mapping):
        raise SanityError(f"{where}.test_cfg must be a table of test-time settings")
    zoo_ap = _decimal(declared.get("zoo_ap", ""), f"{where}.zoo_ap")
    tolerance = _decimal(declared.get("tolerance", ""), f"{where}.tolerance")
    scale = UNITS.get(units)
    if scale is not None:  # ranges are known only once the units are
        if zoo_ap is not None and not Decimal(0) <= zoo_ap <= scale:
            raise SanityError(f"{where}.zoo_ap must be in [0, {scale}] {units}, got {zoo_ap}")
        limit = (cap if cap is not None else max_tolerance()) * scale
        if tolerance is not None and not Decimal(0) <= tolerance <= limit:
            raise SanityError(
                f"{where}.tolerance must be in [0, {limit}] {units} "
                f"({TOOLING_SECTION}.max_tolerance_fraction), got {tolerance}"
            )
    return Expectation(
        detector=detector,
        zoo_ap=zoo_ap,
        tolerance=tolerance,
        units=units,
        source=_text(declared.get("source", ""), f"{where}.source"),
        checkpoint_sha256=_text(
            declared.get("checkpoint_sha256", ""), f"{where}.checkpoint_sha256"
        ),
        image_set=_text(declared.get("image_set", ""), f"{where}.image_set"),
        test_cfg=dict(test_cfg) if test_cfg is not None else None,
        on_fail=on_fail,
    )


@dataclass(frozen=True)
class Configured:
    """What the detector will actually run with, which the declaration must be about."""

    checkpoint_sha256: str
    test_cfg: Mapping[str, Any]


def configured(lab: LabConfig, name: str) -> Configured:
    """The configured checkpoint digest and test-time settings of detector ``name``."""
    settings = detector_settings(lab, name)
    return Configured(str(settings.get("checkpoint_sha256", "")), detector_test_cfg(lab, name))


def needs_declaration(lab: LabConfig, name: str) -> bool:
    """Whether detector ``name`` may run only after Ian's sanity declaration."""
    return str(detector_settings(lab, name)["kind"]) not in EXEMPT_KINDS


def binding_problems(expected: Expectation, configured: Configured) -> list[str]:
    """Where a declaration is about something other than the configured detector.

    Digests are never printed: the check says which one differs, not what it should be.
    """
    where = f"{SECTION}.{expected.detector}"
    problems: list[str] = []
    if expected.checkpoint_sha256 and (
        not configured.checkpoint_sha256
        or expected.checkpoint_sha256 != configured.checkpoint_sha256
    ):
        problems.append(
            f"{where}.checkpoint_sha256 does not match the configured checkpoint_sha256: "
            "the declaration is about another checkpoint"
        )
    if expected.test_cfg is not None and dict(expected.test_cfg) != dict(configured.test_cfg):
        problems.append(
            f"{where}.test_cfg differs from the detector's configured test-time settings "
            f"({sorted(configured.test_cfg)}): the zoo value was measured under other settings"
        )
    return problems


@dataclass
class SanityReport:
    detector: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    #: Whether Ian declared the expectation, bound to this detector: what must precede a run.
    declared: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "verdict": self.verdict,
            "declared": self.declared,
            "reasons": self.reasons,
        }

    def render_text(self) -> str:
        return "; ".join([self.verdict, *self.reasons])


def judge(
    expected: Expectation, measured: object, configured: Configured | None = None
) -> SanityReport:
    """PASS when ``measured`` (pycocotools AP, a fraction) is within the declared tolerance of
    the zoo value, compared in the declared units with the values as written."""
    unbound = binding_problems(expected, configured) if configured is not None else []
    missing = expected.missing() + unbound
    zoo_ap, tolerance, scale = expected.zoo_ap, expected.tolerance, UNITS.get(expected.units)
    if missing or zoo_ap is None or tolerance is None or scale is None:
        return SanityReport(expected.detector, NOT_ASSESSED, missing, declared=False)
    if measured is None:
        return SanityReport(
            expected.detector, NOT_ASSESSED, ["AP measurement is disabled until D8"]
        )
    value = Decimal("NaN")  # anything but a real number is an impossible measurement
    if not isinstance(measured, bool) and isinstance(measured, numbers.Real):
        # float() first: numpy 2 spells repr(np.float64(x)) "np.float64(x)"
        value = Decimal(repr(float(measured)))
    if not (value.is_finite() and Decimal(0) <= value <= Decimal(1)):
        return SanityReport(expected.detector, FAIL, ["the measured AP is not in [0, 1]"])
    if abs(value * scale - zoo_ap) <= tolerance:
        return SanityReport(expected.detector, PASS)
    return SanityReport(
        expected.detector,
        FAIL,
        [
            (
                f"further than {tolerance} {expected.units} from the declared zoo AP "
                f"({expected.source}); on_fail: {expected.on_fail}"
            )
        ],
    )


def measure_ap(detector: str) -> float | None:
    """The detector's raw AP; ``None`` while D8 keeps the measurement disabled."""
    if not AP_MEASUREMENT_ENABLED:
        _log.debug("sanity check for %s: AP measurement disabled until D8", detector)
        return None
    raise NotImplementedError("the pycocotools measurement lands with D8 (PR-F)")


def check_detector(
    detector: str, declared: Mapping[str, Any], configured: Configured | None = None
) -> SanityReport:
    report = judge(expectation(detector, declared), measure_ap(detector), configured)
    _log.info("detector sanity check %s: %s", detector, report.verdict)
    return report


def require_declared(
    detector: str, declared: Mapping[str, Any], configured: Configured
) -> SanityReport:
    """Refuse to run ``detector`` until Ian's declaration is complete and bound to it."""
    report = check_detector(detector, declared, configured)
    if not report.declared:
        raise SanityError(
            f"detector {detector!r} has no complete sanity declaration bound to its "
            "configuration (declared before it runs, D2): " + "; ".join(report.reasons)
        )
    if report.verdict == FAIL:
        raise SanityError(f"detector {detector!r} failed its sanity check: {report.reasons}")
    return report


__all__ = [
    "AP_MEASUREMENT_ENABLED",
    "EXEMPT_KINDS",
    "NOT_ASSESSED",
    "ON_FAIL",
    "UNITS",
    "Configured",
    "Expectation",
    "SanityError",
    "SanityReport",
    "binding_problems",
    "check_detector",
    "configured",
    "expectation",
    "judge",
    "max_tolerance",
    "measure_ap",
    "needs_declaration",
    "require_declared",
]
