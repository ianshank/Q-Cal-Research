"""Calibrators: identity, Platt scaling (arXiv:2405.20459 Eq. 8-9), isotonic regression."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qcal.components import UnknownComponentError
from qcal_lab.calib import CALIBRATORS, build_calibrator, load_calibrator
from qcal_lab.calib.base import CalibrationError, IdentityCalibrator, check_fit_inputs
from qcal_lab.calib.isotonic import (
    IsotonicRegression,
    load_isotonic,
    merge_duplicates,
    pool_adjacent_violators,
)
from qcal_lab.calib.platt import PlattScaling, load_platt, logit, mean_cross_entropy, sigmoid
from qcal_lab.calib.thresholds import select_threshold, threshold_grid
from tests.lab_support import lab_config

unit = st.floats(0.0, 1.0, allow_nan=False)


@pytest.fixture
def lab(tmp_path: Path):
    return lab_config(tmp_path)


def platt(lab) -> PlattScaling:
    return build_calibrator("platt", lab)  # type: ignore[return-value]


def isotonic(lab) -> IsotonicRegression:
    return build_calibrator("isotonic", lab)  # type: ignore[return-value]


# --- inputs and registry -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "targets", "message"),
    [
        ([0.1], [0.1, 0.2], "1 scores but 2 targets"),
        ([], [], "no detections"),
        ([1.5], [0.0], "scores must lie in"),
        ([0.5], [float("nan")], "targets must lie in"),
        ([True], [0.0], "must be numbers"),
    ],
)
def test_fit_inputs_are_checked(scores: list, targets: list, message: str) -> None:
    with pytest.raises(CalibrationError, match=message):
        check_fit_inputs(scores, targets)


def test_registry_and_saved_calibrators(lab) -> None:
    assert CALIBRATORS.names() == ["isotonic", "none", "platt"]
    with pytest.raises(UnknownComponentError):
        build_calibrator("temperature", lab)
    with pytest.raises(CalibrationError, match="string 'name'"):
        load_calibrator({})
    with pytest.raises(CalibrationError, match="expected a saved 'none'"):
        CALIBRATORS.get("none").load({"name": "platt"})


def test_identity(lab) -> None:
    calibrator = build_calibrator("none", lab)
    calibrator.fit([0.2], [0.9])
    assert calibrator.transform([0.2, 0.7]) == [0.2, 0.7]
    assert isinstance(load_calibrator(calibrator.to_dict()), IdentityCalibrator)


# --- Platt scaling -----------------------------------------------------------------------------


def test_platt_recovers_the_generating_parameters(lab) -> None:
    scores = [i / 40 for i in range(1, 40)]
    a, b = 1.7, -0.4
    targets = [sigmoid(a * logit(p, 1e-7) + b) for p in scores]  # soft targets from the model
    model = platt(lab)
    model.fit(scores, targets)
    assert model.converged
    assert (model.a, model.b) == pytest.approx((a, b), abs=1e-6)
    assert model.transform([0.5]) == pytest.approx([sigmoid(b)], abs=1e-6)


def test_platt_reaches_the_minimum_of_the_cross_entropy(lab) -> None:
    scores = [0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1]
    targets = [0.8, 0.0, 0.6, 0.7, 0.1, 0.3, 0.0]
    model = platt(lab)
    model.fit(scores, targets)
    x = [logit(p, 1e-7) for p in scores]
    best = mean_cross_entropy(x, targets, model.a, model.b)
    for da in (-0.05, 0.05):
        for db in (-0.05, 0.05):
            assert mean_cross_entropy(x, targets, max(model.a + da, 0.0), model.b + db) >= best


def test_platt_constraint_a_non_negative_lands_on_the_boundary(lab) -> None:
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    targets = [0.9, 0.7, 0.5, 0.3, 0.1]  # anti-correlated: the free optimum has a < 0
    model = platt(lab)
    model.fit(scores, targets)
    assert model.a == 0.0
    assert model.b == pytest.approx(logit(0.5, 1e-7))
    assert model.transform([0.1, 0.9]) == pytest.approx([0.5, 0.5])


def test_platt_falls_back_to_gradient_steps_when_curvature_is_singular() -> None:
    model = PlattScaling(epsilon=1e-7, max_iterations=200, gradient_tolerance=1e-9, ridge=0.0)
    model.fit([0.4, 0.4, 0.4], [0.2, 0.3, 0.4])  # one distinct score: the Hessian is singular
    assert model.transform([0.4])[0] == pytest.approx(0.3, abs=1e-4)


def test_platt_handles_identical_scores(lab) -> None:
    model = platt(lab)
    model.fit([0.4, 0.4, 0.4], [0.2, 0.3, 0.4])
    assert model.transform([0.4])[0] == pytest.approx(0.3, abs=1e-6)


def test_platt_stops_at_the_iteration_cap(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    lab = lab_config(tmp_path, "[calibrators.platt]\nmax_iterations = 1\n")
    model = platt(lab)
    model.fit([0.1, 0.9, 0.5], [0.0, 1.0, 0.2])
    assert model.iterations == 1
    assert not model.converged
    assert "stopped after 1 iterations" in caplog.text


def test_platt_save_and_load(lab) -> None:
    model = platt(lab)
    model.fit([0.2, 0.8], [0.1, 0.7])
    loaded = load_calibrator(model.to_dict())
    assert loaded.transform([0.3, 0.6]) == model.transform([0.3, 0.6])
    with pytest.raises(CalibrationError, match="a >= 0"):
        load_platt({**model.to_dict(), "a": -1.0})
    with pytest.raises(CalibrationError, match="finite number"):
        load_platt({**model.to_dict(), "b": None})
    with pytest.raises(CalibrationError, match="expected a saved 'platt'"):
        load_platt({"name": "isotonic"})


def test_platt_misuse(lab) -> None:
    with pytest.raises(CalibrationError, match="before fit"):
        platt(lab).transform([0.5])
    with pytest.raises(CalibrationError, match="probability_epsilon"):
        PlattScaling(epsilon=0.6, max_iterations=1, gradient_tolerance=1, ridge=0)
    with pytest.raises(CalibrationError, match="positive"):
        PlattScaling(epsilon=0.1, max_iterations=0, gradient_tolerance=1, ridge=0)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(unit, unit), min_size=2, max_size=30))
def test_platt_is_monotone_and_bounded(pairs: list[tuple[float, float]]) -> None:
    model = PlattScaling(epsilon=1e-7, max_iterations=100, gradient_tolerance=1e-9, ridge=1e-12)
    model.fit([p for p, _ in pairs], [t for _, t in pairs])
    assert model.a >= 0.0
    grid = [i / 20 for i in range(21)]
    out = model.transform(grid)
    assert all(0.0 <= v <= 1.0 for v in out)
    assert all(b >= a - 1e-12 for a, b in itertools.pairwise(out))


# --- isotonic regression -----------------------------------------------------------------------


def test_pava_pools_violators_with_weights() -> None:
    assert pool_adjacent_violators([1.0, 3.0, 2.0, 4.0], [1, 1, 1, 1]) == [1.0, 2.5, 2.5, 4.0]
    assert pool_adjacent_violators([3.0, 1.0], [1, 3]) == [1.5, 1.5]
    with pytest.raises(CalibrationError, match="positive"):
        pool_adjacent_violators([1.0], [0.0])
    with pytest.raises(CalibrationError, match="differ in length"):
        pool_adjacent_violators([1.0], [])


def test_duplicate_scores_merge_into_their_mean() -> None:
    xs, ys, counts = merge_duplicates([0.5, 0.2, 0.5, 0.5 + 1e-16], [1.0, 0.0, 0.0, 0.5])
    assert xs == [0.2, 0.5]
    assert ys == pytest.approx([0.0, 0.5])
    assert counts == [1.0, 3.0]


def test_isotonic_interpolates_and_clips(lab) -> None:
    model = isotonic(lab)
    model.fit([0.2, 0.4, 0.6, 0.8], [0.1, 0.5, 0.3, 0.9])
    assert model.y == pytest.approx([0.1, 0.4, 0.4, 0.9])
    assert model.transform([0.2, 0.3, 0.7, 0.0, 1.0]) == pytest.approx([0.1, 0.25, 0.65, 0.1, 0.9])


def test_isotonic_single_knot_and_bounds(tmp_path: Path) -> None:
    lab = lab_config(tmp_path, "[calibrators.isotonic]\ny_min = 0.2\ny_max = 0.6\n")
    model = isotonic(lab)
    model.fit([0.5, 0.5], [0.0, 0.1])
    assert model.transform([0.1, 0.9]) == [0.2, 0.2]
    model.fit([0.1, 0.9], [0.0, 1.0])
    assert model.y == [0.2, 0.6]


def test_isotonic_out_of_bounds_raise(tmp_path: Path) -> None:
    lab = lab_config(tmp_path, '[calibrators.isotonic]\nout_of_bounds = "raise"\n')
    model = isotonic(lab)
    model.fit([0.2, 0.8], [0.1, 0.9])
    with pytest.raises(CalibrationError, match="outside the fitted range"):
        model.transform([0.9])


def test_isotonic_save_load_and_misuse(lab) -> None:
    model = isotonic(lab)
    with pytest.raises(CalibrationError, match="before fit"):
        model.transform([0.5])
    model.fit([0.2, 0.8], [0.1, 0.9])
    assert load_calibrator(model.to_dict()).transform([0.5]) == model.transform([0.5])
    saved = model.to_dict()
    for broken, message in [
        ({**saved, "x": []}, "non-empty"),
        ({**saved, "y": [0.9, 0.1]}, "non-decreasing"),
        ({**saved, "out_of_bounds": None}, "out_of_bounds"),
        ({"name": "platt"}, "expected a saved 'isotonic'"),
    ]:
        with pytest.raises(CalibrationError, match=message):
            load_isotonic(broken)
    with pytest.raises(CalibrationError, match="out_of_bounds must be"):
        IsotonicRegression(y_min=0, y_max=1, out_of_bounds="nan")
    with pytest.raises(CalibrationError, match="bounds"):
        IsotonicRegression(y_min=0.5, y_max=0.1, out_of_bounds="clip")
    with pytest.raises(CalibrationError, match="knots differ"):
        IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", x=[0.1], y=[])


@settings(max_examples=80, deadline=None)
@given(st.lists(st.tuples(unit, unit), min_size=1, max_size=40))
def test_isotonic_properties(pairs: list[tuple[float, float]]) -> None:
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    scores, targets = [p for p, _ in pairs], [t for _, t in pairs]
    model.fit(scores, targets)
    assert model.y == sorted(model.y)  # non-decreasing
    assert min(targets) - 1e-12 <= model.y[0]
    assert model.y[-1] <= max(targets) + 1e-12
    _, _, counts = merge_duplicates(scores, targets)
    # Pooling preserves the total: the weighted fit sums to the sum of targets.
    assert math.fsum(y * c for y, c in zip(model.y, counts, strict=True)) == pytest.approx(
        math.fsum(targets), abs=1e-9
    )
    out = model.transform(sorted(scores))
    assert out == sorted(out)


# --- thresholds -------------------------------------------------------------------------------


def test_threshold_grid() -> None:
    assert threshold_grid(0.0, 0.3, 0.1) == (0.0, 0.1, 0.2, 0.3)
    assert threshold_grid(0.5, 0.5, 0.1) == (0.5,)
    assert len(threshold_grid(0.0, 0.95, 0.05)) == 20
    for bad in [(0.0, 1.0, 0.0), (0.6, 0.5, 0.1), (0.0, 1.5, 0.1), (float("nan"), 1.0, 0.1)]:
        with pytest.raises(CalibrationError):
            threshold_grid(*bad)


def test_select_threshold() -> None:
    values = {0.0: 3.0, 0.1: 1.0, 0.2: 1.0, 0.3: float("nan")}
    assert select_threshold(list(values), values.__getitem__) == (0.1, False)  # ties: earliest
    assert select_threshold([0.0, 0.5], lambda _t: float("nan")) == (0.0, True)
    with pytest.raises(CalibrationError, match="no threshold candidates"):
        select_threshold([], lambda _t: 0.0)
    with pytest.raises(CalibrationError, match="not a number"):
        select_threshold([0.0], lambda _t: None)  # type: ignore[arg-type,return-value]
