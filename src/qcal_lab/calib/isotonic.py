"""Isotonic regression by pool-adjacent-violators, following scikit-learn's conventions.

Kuzucu et al. (arXiv:2405.20459, Sec. 4.4) fit scikit-learn's ``IsotonicRegression`` on
(score, IoU) pairs. This clean-room version follows that estimator's documented behaviour,
so the parity tests can compare its outputs directly:

- duplicate scores are merged into their weighted mean target before pooling;
- the fit is increasing and then clipped to ``[y_min, y_max]``;
- prediction interpolates linearly between fitted points. Outside the fitted range it clips,
  or it raises with ``out_of_bounds = "raise"``.

Duplicate scores are merged when they differ by less than float64 resolution (1e-15). This
detail of scikit-learn's ``_make_unique`` is reproduced from its source as remembered
[unverified]; the parity tests against the oracle decide.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from typing import Any, Final

from qcal_lab.calib.base import (
    CALIBRATORS,
    CalibrationError,
    CalibratorKind,
    check_fit_inputs,
    check_scores,
    finite_number,
)
from qcal_lab.config import LabConfig

OUT_OF_BOUNDS: Final = frozenset({"clip", "raise"})
_RESOLUTION: Final = 1e-15


def pool_adjacent_violators(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """The weighted least-squares non-decreasing fit to ``values`` (in their given order)."""
    if len(values) != len(weights):
        raise CalibrationError("values and weights differ in length")
    if any(w <= 0 for w in weights):
        raise CalibrationError("weights must be positive")
    blocks: list[list[float]] = []  # [weighted sum, total weight, count]
    for value, weight in zip(values, weights, strict=True):
        blocks.append([value * weight, weight, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, weight_sum, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += weight_sum
            blocks[-1][2] += count
    fitted: list[float] = []
    for total, weight_sum, count in blocks:
        fitted.extend([total / weight_sum] * int(count))
    return fitted


def merge_duplicates(
    x: Sequence[float], y: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    """Sorted unique x, the mean y at each, and the number of samples merged into each."""
    order = sorted(range(len(x)), key=lambda i: (x[i], y[i]))
    ux: list[float] = []
    sums: list[float] = []
    counts: list[float] = []
    for i in order:
        if ux and x[i] - ux[-1] < _RESOLUTION:
            sums[-1] += y[i]
            counts[-1] += 1.0
        else:
            ux.append(x[i])
            sums.append(y[i])
            counts.append(1.0)
    return ux, [s / c for s, c in zip(sums, counts, strict=True)], counts


class IsotonicRegression:
    name = "isotonic"

    def __init__(
        self,
        *,
        y_min: float,
        y_max: float,
        out_of_bounds: str,
        x: Sequence[float] = (),
        y: Sequence[float] = (),
    ) -> None:
        if out_of_bounds not in OUT_OF_BOUNDS:
            raise CalibrationError(
                f"out_of_bounds must be one of {sorted(OUT_OF_BOUNDS)}, got {out_of_bounds!r}"
            )
        if not 0.0 <= y_min <= y_max <= 1.0:
            raise CalibrationError("isotonic bounds must satisfy 0 <= y_min <= y_max <= 1")
        if len(x) != len(y):
            raise CalibrationError("isotonic knots differ in length")
        self.y_min, self.y_max, self.out_of_bounds = y_min, y_max, out_of_bounds
        self.x, self.y = list(x), list(y)

    @classmethod
    def from_config(cls, lab: LabConfig) -> IsotonicRegression:
        cfg = lab.config
        return cls(
            y_min=cfg.float_value("calibrators.isotonic.y_min"),
            y_max=cfg.float_value("calibrators.isotonic.y_max"),
            out_of_bounds=cfg.str_value("calibrators.isotonic.out_of_bounds"),
        )

    def fit(self, scores: Sequence[float], targets: Sequence[float]) -> None:
        x, y = check_fit_inputs(scores, targets)
        ux, uy, counts = merge_duplicates(x, y)
        fitted = pool_adjacent_violators(uy, counts)
        self.x = ux
        self.y = [min(max(v, self.y_min), self.y_max) for v in fitted]

    def _value(self, p: float) -> float:
        lo, hi = self.x[0], self.x[-1]
        if p < lo or p > hi:
            if self.out_of_bounds == "raise":
                raise CalibrationError(f"score {p} is outside the fitted range [{lo}, {hi}]")
            p = min(max(p, lo), hi)
        right = bisect.bisect_left(self.x, p)
        if right < len(self.x) and self.x[right] == p:
            return self.y[right]
        left = right - 1
        x0, x1, y0, y1 = self.x[left], self.x[right], self.y[left], self.y[right]
        return y0 + (y1 - y0) * (p - x0) / (x1 - x0)

    def transform(self, scores: Sequence[float]) -> list[float]:
        if not self.x:
            raise CalibrationError("isotonic regression used before fit")
        return [self._value(p) for p in check_scores(scores)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": list(self.x),
            "y": list(self.y),
            "y_min": self.y_min,
            "y_max": self.y_max,
            "out_of_bounds": self.out_of_bounds,
        }


def load_isotonic(data: Mapping[str, Any]) -> IsotonicRegression:
    if data.get("name") != IsotonicRegression.name:
        raise CalibrationError(f"expected a saved 'isotonic' calibrator, got {data.get('name')!r}")
    knots_x, knots_y = data.get("x"), data.get("y")
    if not isinstance(knots_x, list) or not isinstance(knots_y, list) or not knots_x:
        raise CalibrationError("a saved isotonic calibrator needs non-empty 'x' and 'y' lists")
    xs = check_scores(knots_x)
    ys = check_scores(knots_y)
    if xs != sorted(xs) or ys != sorted(ys):
        raise CalibrationError("saved isotonic knots must be non-decreasing")
    out_of_bounds = data.get("out_of_bounds")
    if not isinstance(out_of_bounds, str):
        raise CalibrationError("a saved isotonic calibrator needs 'out_of_bounds'")
    return IsotonicRegression(
        y_min=finite_number(data, "y_min"),
        y_max=finite_number(data, "y_max"),
        out_of_bounds=out_of_bounds,
        x=xs,
        y=ys,
    )


CALIBRATORS.register(
    IsotonicRegression.name,
    CalibratorKind(build=IsotonicRegression.from_config, load=load_isotonic),
)

__all__ = [
    "IsotonicRegression",
    "load_isotonic",
    "merge_duplicates",
    "pool_adjacent_violators",
]
