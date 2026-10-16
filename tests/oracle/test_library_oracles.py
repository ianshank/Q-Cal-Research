"""Our calibrators against scikit-learn and SciPy on the same inputs (TD-13, TD-16)."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from qcal_lab.calib.isotonic import IsotonicRegression
from qcal_lab.calib.platt import PlattScaling
from tests.oracle.conftest import oracle
from tests.reference.calibration import logit, platt_gradient, platt_loss

pytestmark = pytest.mark.rule("C3")

EPSILON = 1e-7
QUERIES = [k / 200 for k in range(201)]


def tie_heavy_sample(seed: int, n: int) -> tuple[list[float], list[float]]:
    """Scores on a coarse grid plus neighbours one or two ulps away: exact and near ties."""
    rng = random.Random(seed)  # noqa: S311 - reproducible test data
    scores, targets = [], []
    for _ in range(n):
        base = rng.randrange(1, 40) / 40
        score = base
        for _ in range(rng.choice([0, 0, 1, 2])):
            score = math.nextafter(score, 1.0)
        scores.append(score)
        targets.append(rng.random() if rng.random() < 0.8 else 0.0)
    return scores, targets


def ours_isotonic(
    scores: list[float], targets: list[float], resolution: float
) -> IsotonicRegression:
    model = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip", duplicate_resolution=resolution
    )
    model.fit(scores, targets)
    return model


@pytest.mark.parametrize("seed", range(20))
def test_isotonic_matches_scikit_learn_float64(seed: int) -> None:
    np = oracle("numpy")
    isotonic = oracle("sklearn.isotonic")
    scores, targets = tie_heavy_sample(seed, 300)
    reference = isotonic.IsotonicRegression(
        increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip"
    ).fit(np.asarray(scores, dtype=np.float64), np.asarray(targets, dtype=np.float64))
    expected = reference.predict(np.asarray(QUERIES + scores, dtype=np.float64)).tolist()
    actual = ours_isotonic(scores, targets, 1e-15).transform(QUERIES + scores)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_isotonic_matches_scikit_learn_float32(seed: int) -> None:
    """With float32 inputs scikit-learn merges scores within 1e-6 (the dtype's resolution)."""
    np = oracle("numpy")
    isotonic = oracle("sklearn.isotonic")
    scores, targets = tie_heavy_sample(seed, 300)
    x32 = np.asarray(scores, dtype=np.float32)
    reference = isotonic.IsotonicRegression(
        increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip"
    ).fit(x32, np.asarray(targets, dtype=np.float32))
    as_float32 = [float(v) for v in x32]
    expected = reference.predict(np.asarray(QUERIES, dtype=np.float32)).tolist()
    actual = ours_isotonic(as_float32, targets, 1e-6).transform(QUERIES)
    assert actual == pytest.approx(expected, abs=2e-6)


def _lbfgsb(x: list[float], y: list[float]) -> tuple[float, float]:
    optimize = oracle("scipy.optimize")

    def loss(params: Any) -> float:
        return platt_loss(x, y, float(params[0]), float(params[1]))

    def gradient(params: Any) -> list[float]:
        return list(platt_gradient(x, y, float(params[0]), float(params[1])))

    result = optimize.minimize(
        loss,
        x0=[1.0, 0.0],
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0.0, None), (None, None)],
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
    )
    return float(result.x[0]), float(result.x[1])


@pytest.mark.parametrize("seed", range(20))
def test_platt_reaches_the_l_bfgs_b_optimum(seed: int) -> None:
    """Compare losses and calibrated outputs, not raw (a, b): flat directions make the
    parameters less well determined than what they predict."""
    rng = random.Random(seed)  # noqa: S311 - reproducible test data
    scores = [rng.randrange(1, 40) / 40 for _ in range(200)]
    noise = rng.uniform(0.0, 0.3)
    targets = [min(max(s + rng.gauss(0, noise) - 0.1, 0.0), 1.0) for s in scores]
    if seed % 5 == 0:  # some anti-correlated sets, whose optimum is on the boundary a = 0
        targets = [1.0 - t for t in targets]
    model = PlattScaling(
        epsilon=EPSILON,
        max_iterations=100,
        gradient_tolerance=1e-9,
        ridge=1e-12,
        armijo=1e-4,
        min_step=1e-12,
    )
    model.fit(scores, targets)
    x = [logit(p, EPSILON) for p in scores]
    a_ref, b_ref = _lbfgsb(x, targets)
    ours, theirs = platt_loss(x, targets, model.a, model.b), platt_loss(x, targets, a_ref, b_ref)
    assert ours <= theirs + 1e-10
    assert ours == pytest.approx(theirs, abs=1e-10)
    reference = PlattScaling(
        epsilon=EPSILON, max_iterations=1, gradient_tolerance=1.0, ridge=0.0, a=a_ref, b=b_ref,
        fitted=True,
    )  # fmt: skip
    assert model.transform(QUERIES) == pytest.approx(reference.transform(QUERIES), abs=1e-6)


def test_pycocotools_reads_the_fixture_like_our_loader(tmp_path: Any) -> None:
    """The synthetic fixture is valid COCO for the reference toolkit too."""
    coco_module = oracle("pycocotools.coco")
    from qcal_lab.data.coco import load_coco
    from qcal_lab.data.fixture import write_fixture
    from tests.lab_support import fixture_document

    path = tmp_path / "gt.json"
    write_fixture(path, fixture_document())
    reference = coco_module.COCO(str(path))
    ours = load_coco(path)
    assert sorted(str(i) for i in reference.getImgIds()) == sorted(ours.images)
    assert len(reference.getAnnIds()) == sum(len(v) for v in ours.boxes.values())
    assert sorted(reference.getCatIds()) == sorted(ours.category_ids)
