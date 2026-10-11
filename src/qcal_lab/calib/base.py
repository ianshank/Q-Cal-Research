"""Calibrator registry, input validation and the identity calibrator.

Every calibrator implements :class:`qcal.protocols.Calibrator` and can be saved with
:func:`save_calibrator` (its ``to_dict`` in a format envelope) and restored with
:func:`load_calibrator`. A fitted calibrator is stored as a run artifact, so the registry
records exactly what was applied to the test predictions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

from qcal.components import ComponentRegistry
from qcal.protocols import Calibrator
from qcal_lab.config import LabConfig
from qcal_lab.formats import envelope, open_envelope

CALIBRATOR_FORMAT: Final = "qcal_lab.calibrator"
CALIBRATOR_VERSION: Final = 1


class CalibrationError(ValueError):
    """Calibrator inputs are malformed, or a calibrator was used before fitting."""


@runtime_checkable
class SerializableCalibrator(Calibrator, Protocol):
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CalibratorKind:
    build: Callable[[LabConfig], SerializableCalibrator]
    load: Callable[[Mapping[str, Any]], SerializableCalibrator]
    #: Whether fitting reads the fit data. A kind that ignores it (the identity) makes the
    #: fit draw, and so the seed, change nothing.
    uses_fit_data: bool = True


CALIBRATORS: ComponentRegistry[CalibratorKind] = ComponentRegistry("calibrator")


def _unit_interval(values: Sequence[float], what: str) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CalibrationError(f"{what} must be numbers, got {value!r}")
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise CalibrationError(f"{what} must lie in [0, 1], got {value!r}")
        out.append(float(value))
    return out


def check_scores(scores: Sequence[float]) -> list[float]:
    return _unit_interval(scores, "scores")


def check_fit_inputs(
    scores: Sequence[float], targets: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Scores and targets of equal, non-zero length, all in [0, 1]."""
    if len(scores) != len(targets):
        raise CalibrationError(f"{len(scores)} scores but {len(targets)} targets")
    if not scores:
        raise CalibrationError("cannot fit a calibrator on no detections")
    return check_scores(scores), _unit_interval(targets, "targets")


class IdentityCalibrator:
    """Leaves scores unchanged: the ``none`` calibrator, and the fallback for empty classes."""

    name = "none"

    def fit(self, scores: Sequence[float], targets: Sequence[float]) -> None:
        check_fit_inputs(scores, targets)

    def transform(self, scores: Sequence[float]) -> list[float]:
        return check_scores(scores)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


def _expect_name(data: Mapping[str, Any], name: str) -> None:
    if data.get("name") != name:
        raise CalibrationError(f"expected a saved {name!r} calibrator, got {data.get('name')!r}")


def _load_identity(data: Mapping[str, Any]) -> IdentityCalibrator:
    _expect_name(data, IdentityCalibrator.name)
    return IdentityCalibrator()


CALIBRATORS.register(
    IdentityCalibrator.name,
    CalibratorKind(
        build=lambda _lab: IdentityCalibrator(), load=_load_identity, uses_fit_data=False
    ),
)


def build_calibrator(name: str, lab: LabConfig) -> SerializableCalibrator:
    return CALIBRATORS.get(name).build(lab)


def save_calibrator(calibrator: SerializableCalibrator) -> dict[str, Any]:
    """A calibrator as a saved document: its ``to_dict`` with format and version."""
    return envelope(CALIBRATOR_FORMAT, CALIBRATOR_VERSION, calibrator.to_dict())


def load_calibrator(data: Mapping[str, Any]) -> SerializableCalibrator:
    """Restore a :func:`save_calibrator` document; one without its envelope is refused."""
    body = open_envelope(
        data, CALIBRATOR_FORMAT, CALIBRATOR_VERSION, "a saved calibrator", error=CalibrationError
    )
    name = body.get("name")
    if not isinstance(name, str):
        raise CalibrationError("a saved calibrator needs a string 'name'")
    return CALIBRATORS.get(name).load(body)


def finite_number(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CalibrationError(f"saved calibrator field {key!r} must be a finite number")
    return float(value)


def saved_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError(f"saved calibrator field {key!r} must be an integer")
    return value


def saved_flag(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise CalibrationError(f"saved calibrator field {key!r} must be true or false")
    return value


__all__ = [
    "CALIBRATORS",
    "CALIBRATOR_FORMAT",
    "CALIBRATOR_VERSION",
    "CalibrationError",
    "CalibratorKind",
    "IdentityCalibrator",
    "SerializableCalibrator",
    "build_calibrator",
    "check_fit_inputs",
    "check_scores",
    "finite_number",
    "load_calibrator",
    "save_calibrator",
    "saved_flag",
    "saved_int",
]
