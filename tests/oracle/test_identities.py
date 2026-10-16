"""In-sample identities that hold for any data; they need no library."""

from __future__ import annotations

import random

import pytest

from qcal_lab.calib.isotonic import IsotonicRegression
from qcal_lab.calib.platt import PlattScaling

pytestmark = pytest.mark.rule("C3")


def sample(seed: int) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)  # noqa: S311 - reproducible test data
    scores = [rng.randrange(1, 40) / 40 for _ in range(150)]
    targets = [min(max(s * rng.uniform(0.5, 1.2) + rng.gauss(0, 0.1), 0.0), 1.0) for s in scores]
    return scores, targets


def platt() -> PlattScaling:
    return PlattScaling(
        epsilon=1e-7, max_iterations=100, gradient_tolerance=1e-9, ridge=1e-12,
        armijo=1e-4, min_step=1e-12,
    )  # fmt: skip


def squared_error(predictions: list[float], targets: list[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(predictions, targets, strict=True))


@pytest.mark.parametrize("seed", range(15))
def test_an_interior_platt_optimum_matches_the_mean_target(seed: int) -> None:
    """dL/db = mean(sigmoid(.) - y) = 0 at an interior optimum."""
    scores, targets = sample(seed)
    model = platt()
    model.fit(scores, targets)
    assert model.a > 0
    assert model.converged
    calibrated = model.transform(scores)
    assert sum(calibrated) / len(calibrated) == pytest.approx(sum(targets) / len(targets), abs=1e-8)


@pytest.mark.parametrize("seed", range(15))
def test_isotonic_has_the_least_squared_error_among_monotone_maps(seed: int) -> None:
    """Platt (a >= 0) and the identity are monotone, so they lie in isotonic's feasible set."""
    scores, targets = sample(seed)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(scores, targets)
    model = platt()
    model.fit(scores, targets)
    best = squared_error(iso.transform(scores), targets)
    assert best <= squared_error(model.transform(scores), targets) + 1e-12
    assert best <= squared_error(scores, targets) + 1e-12
