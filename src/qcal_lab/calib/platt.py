"""Platt scaling for object detection (Kuzucu et al., arXiv:2405.20459, Eq. 8 and 9).

    p_cal = sigmoid(a * logit(p) + b),  a >= 0

fit by minimising the mean cross-entropy against soft targets: the IoU of each detection with
its matched object, and 0 for a false positive (App. C.3, Eq. A.39 to A.41). The paper uses
L-BFGS. The objective is convex in (a, b), so this damped Newton method with an Armijo
backtracking line search reaches the same minimiser. When the unconstrained minimiser has
a < 0, the constrained one lies on the boundary a = 0. There the loss depends on b alone and
is minimised at sigmoid(b) = mean target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from qcal.log import get_logger
from qcal_lab.calib.base import (
    CALIBRATORS,
    CalibrationError,
    CalibratorKind,
    check_fit_inputs,
    check_scores,
    finite_number,
)
from qcal_lab.config import LabConfig

_log = get_logger("lab.calib.platt")


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


#: Below this many units in the last place of the loss, a predicted decrease is rounding noise.
_RESOLVABLE_ULPS: Final = 16


def softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def logit(p: float, epsilon: float) -> float:
    clipped = min(max(p, epsilon), 1.0 - epsilon)
    return math.log(clipped) - math.log1p(-clipped)


def mean_cross_entropy(x: Sequence[float], y: Sequence[float], a: float, b: float) -> float:
    """Mean of -(y log s + (1 - y) log(1 - s)) with s = sigmoid(a x + b), computed stably."""
    terms = (softplus(a * xi + b) - yi * (a * xi + b) for xi, yi in zip(x, y, strict=True))
    return math.fsum(terms) / len(x)


class PlattScaling:
    name = "platt"

    def __init__(
        self,
        *,
        epsilon: float,
        max_iterations: int,
        gradient_tolerance: float,
        ridge: float,
        armijo: float = 1e-4,
        min_step: float = 1e-12,
        a: float = 1.0,
        b: float = 0.0,
        fitted: bool = False,
    ) -> None:
        if not 0.0 < epsilon < 0.5:
            raise CalibrationError("probability_epsilon must lie in (0, 0.5)")
        if max_iterations <= 0 or gradient_tolerance <= 0 or ridge < 0:
            raise CalibrationError("Platt settings must be positive")
        if not (0.0 < armijo < 0.5 and 0.0 < min_step < 1.0):
            raise CalibrationError("Platt line search needs 0 < armijo < 0.5 and 0 < min_step < 1")
        self.armijo, self.min_step = armijo, min_step
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.gradient_tolerance = gradient_tolerance
        self.ridge = ridge
        self.a, self.b = a, b
        self.fitted = fitted
        self.iterations = 0
        self.converged = fitted

    @classmethod
    def from_config(cls, lab: LabConfig) -> PlattScaling:
        cfg = lab.config
        return cls(
            epsilon=cfg.float_value("calibrators.probability_epsilon"),
            max_iterations=cfg.int_value("calibrators.platt.max_iterations"),
            gradient_tolerance=cfg.float_value("calibrators.platt.gradient_tolerance"),
            ridge=cfg.float_value("calibrators.platt.ridge"),
            armijo=cfg.float_value("calibrators.platt.armijo"),
            min_step=cfg.float_value("calibrators.platt.min_step"),
        )

    def _newton_direction(
        self, x: Sequence[float], y: Sequence[float], a: float, b: float
    ) -> tuple[float, float, float, float]:
        """Gradient (ga, gb) and Newton step (da, db) of the mean cross-entropy at (a, b)."""
        n = len(x)
        ga = gb = haa = hab = hbb = 0.0
        for xi, yi in zip(x, y, strict=True):
            s = sigmoid(a * xi + b)
            r, w = s - yi, s * (1.0 - s)
            ga += r * xi
            gb += r
            haa += w * xi * xi
            hab += w * xi
            hbb += w
        ga, gb = ga / n, gb / n
        haa, hab, hbb = haa / n + self.ridge, hab / n, hbb / n + self.ridge
        det = haa * hbb - hab * hab
        if det > 0.0 and haa > 0.0:
            return ga, gb, (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        # Singular curvature (for example every score identical): fall back to gradient descent.
        return ga, gb, ga, gb

    def fit(self, scores: Sequence[float], targets: Sequence[float]) -> None:
        probabilities, y = check_fit_inputs(scores, targets)
        x = [logit(p, self.epsilon) for p in probabilities]
        a, b = 1.0, 0.0
        loss = mean_cross_entropy(x, y, a, b)
        self.converged = False
        for iteration in range(1, self.max_iterations + 1):
            self.iterations = iteration
            ga, gb, da, db = self._newton_direction(x, y, a, b)
            if max(abs(ga), abs(gb)) < self.gradient_tolerance:
                self.converged = True
                break
            decrease = ga * da + gb * db
            if decrease <= _RESOLVABLE_ULPS * math.ulp(loss):
                # The predicted decrease is below what the loss can resolve, so the Armijo test
                # would only see rounding and accept or refuse tiny steps at random. This close
                # to the optimum of a convex loss the full Newton step converges on its own.
                a, b = a - da, b - db
                loss = mean_cross_entropy(x, y, a, b)
                continue
            step = 1.0
            while True:
                na, nb = a - step * da, b - step * db
                new_loss = mean_cross_entropy(x, y, na, nb)
                if new_loss <= loss - self.armijo * step * decrease or step < self.min_step:
                    break
                step /= 2.0
            if step < self.min_step:
                _log.warning("Platt line search found no decrease at iteration %d", iteration)
                break
            a, b, loss = na, nb, new_loss
        if not self.converged:
            _log.warning("Platt scaling stopped unconverged after %d iteration(s)", self.iterations)
        if a < 0.0:  # the boundary solution is exact only at a converged unconstrained optimum
            a, b = 0.0, logit(math.fsum(y) / len(y), self.epsilon)
        self.a, self.b, self.fitted = a, b, True

    def transform(self, scores: Sequence[float]) -> list[float]:
        if not self.fitted:
            raise CalibrationError("Platt scaling used before fit")
        return [sigmoid(self.a * logit(p, self.epsilon) + self.b) for p in check_scores(scores)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "a": self.a,
            "b": self.b,
            "epsilon": self.epsilon,
            "iterations": self.iterations,
            "converged": self.converged,
        }


def load_platt(data: Mapping[str, Any]) -> PlattScaling:
    if data.get("name") != PlattScaling.name:
        raise CalibrationError(f"expected a saved 'platt' calibrator, got {data.get('name')!r}")
    a = finite_number(data, "a")
    if a < 0:
        raise CalibrationError("a saved Platt calibrator must have a >= 0")
    return PlattScaling(
        epsilon=finite_number(data, "epsilon"),
        max_iterations=1,
        gradient_tolerance=1.0,
        ridge=0.0,
        a=a,
        b=finite_number(data, "b"),
        fitted=True,
    )


CALIBRATORS.register(
    PlattScaling.name, CalibratorKind(build=PlattScaling.from_config, load=load_platt)
)

__all__ = ["PlattScaling", "load_platt", "logit", "mean_cross_entropy", "sigmoid", "softplus"]
