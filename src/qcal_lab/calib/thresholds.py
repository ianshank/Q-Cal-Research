"""Threshold grids and selection by an objective (lower is better)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

from qcal_lab.calib.base import CalibrationError

_DIGITS: Final = 12  # grid points are rounded so 0.1 + 0.2 style drift cannot change them


def threshold_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Evenly spaced thresholds from ``start`` to ``stop`` inclusive, all in [0, 1]."""
    if not (math.isfinite(start) and math.isfinite(stop) and math.isfinite(step)):
        raise CalibrationError("threshold grid bounds must be finite")
    if step <= 0 or not 0.0 <= start <= stop <= 1.0:
        raise CalibrationError("threshold grid needs 0 <= start <= stop <= 1 and step > 0")
    count = math.floor((stop - start) / step + 1e-9) + 1
    return tuple(round(start + i * step, _DIGITS) for i in range(count))


def select_threshold(
    candidates: Sequence[float], objective: Callable[[float], float]
) -> tuple[float, bool]:
    """The candidate with the lowest finite objective; ties keep the earliest candidate.

    Returns ``(threshold, fell_back)``: when no candidate has a finite objective (for example
    a class with no detections and no objects), the first candidate is used and
    ``fell_back`` is true so the run's calibration artifact records it.
    """
    if not candidates:
        raise CalibrationError("no threshold candidates")
    best: tuple[float, float] | None = None
    for threshold in candidates:
        value = objective(threshold)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CalibrationError(f"threshold objective returned {value!r}, not a number")
        if not math.isfinite(value):
            continue
        if best is None or value < best[1]:
            best = (threshold, float(value))
    if best is None:
        return candidates[0], True
    return best[0], False


__all__ = ["select_threshold", "threshold_grid"]
