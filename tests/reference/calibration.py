"""Reference implementations for the calibrators and the two-threshold procedure.

Each one is written from the mathematical definition, not from the production code, and is
deliberately slow (cubic where the production code is linear) so the two cannot share a bug:

- isotonic regression by the min-max formula ``f_i = max_{j<=i} min_{k>=i} mean(y[j..k])``
  (weighted), which characterises the least-squares non-decreasing fit;
- the Platt cross-entropy and its gradient written out term by term;
- threshold selection as "the earliest candidate with the smallest finite objective";
- Alg. A.2 as three explicit passes: keep >= u_c, transform, keep >= v_c.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from qcal.protocols import Detection, ImageDetections


def merge_exact(
    x: Sequence[float], y: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    """Unique x (sorted), mean y at each, and the count merged into each (exact equality)."""
    groups: dict[float, list[float]] = {}
    for xi, yi in zip(x, y, strict=True):
        groups.setdefault(xi, []).append(yi)
    keys = sorted(groups)
    return (
        keys,
        [sum(groups[k]) / len(groups[k]) for k in keys],
        [float(len(groups[k])) for k in keys],
    )


def isotonic_minmax(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """The weighted least-squares non-decreasing fit, by the min-max formula (O(n^3))."""
    n = len(values)

    def mean(j: int, k: int) -> float:
        w = sum(weights[j : k + 1])
        return sum(v * wt for v, wt in zip(values[j : k + 1], weights[j : k + 1], strict=True)) / w

    return [max(min(mean(j, k) for k in range(i, n)) for j in range(i + 1)) for i in range(n)]


def isotonic_reference(x: Sequence[float], y: Sequence[float]) -> tuple[list[float], list[float]]:
    ux, uy, counts = merge_exact(x, y)
    return ux, isotonic_minmax(uy, counts)


def interpolate(knots_x: Sequence[float], knots_y: Sequence[float], p: float) -> float:
    """Linear interpolation between knots, clipped to the end values outside them."""
    if p <= knots_x[0]:
        return knots_y[0]
    if p >= knots_x[-1]:
        return knots_y[-1]
    for i in range(len(knots_x) - 1):
        if knots_x[i] <= p <= knots_x[i + 1]:
            if p == knots_x[i]:
                return knots_y[i]
            t = (p - knots_x[i]) / (knots_x[i + 1] - knots_x[i])
            return knots_y[i] + t * (knots_y[i + 1] - knots_y[i])
    raise AssertionError("unreachable")


def logit(p: float, epsilon: float) -> float:
    q = min(max(p, epsilon), 1.0 - epsilon)
    return math.log(q / (1.0 - q))


def platt_loss(x: Sequence[float], y: Sequence[float], a: float, b: float) -> float:
    """Mean cross-entropy of sigmoid(a x + b) against soft targets y, term by term."""
    total = 0.0
    for xi, yi in zip(x, y, strict=True):
        z = a * xi + b
        # -(y log s + (1 - y) log(1 - s)) = log(1 + e^z) - y z
        total += (z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))) - yi * z
    return total / len(x)


def platt_gradient(
    x: Sequence[float], y: Sequence[float], a: float, b: float
) -> tuple[float, float]:
    ga = gb = 0.0
    for xi, yi in zip(x, y, strict=True):
        s = 1.0 / (1.0 + math.exp(-(a * xi + b)))
        ga += (s - yi) * xi
        gb += s - yi
    return ga / len(x), gb / len(x)


def select_reference(
    candidates: Sequence[float], objective: Callable[[float], float]
) -> tuple[float, bool]:
    values = [objective(c) for c in candidates]
    finite = [(v, i) for i, v in enumerate(values) if math.isfinite(v)]
    if not finite:
        return candidates[0], True
    best = min(v for v, _ in finite)
    return candidates[min(i for v, i in finite if v == best)], False


def apply_reference(
    images: Sequence[ImageDetections],
    u: Mapping[int, float],
    v: Mapping[int, float],
    transform: Callable[[int, float], float],
) -> list[ImageDetections]:
    """Alg. A.2 as three passes over each image."""
    out = []
    for image in images:
        kept = [d for d in image.detections if d.score >= u[d.label]]
        calibrated = [
            Detection(d.box_xyxy, transform(d.label, d.score), d.label, d.logit) for d in kept
        ]
        out.append(
            ImageDetections(image.image_id, tuple(d for d in calibrated if d.score >= v[d.label]))
        )
    return out
