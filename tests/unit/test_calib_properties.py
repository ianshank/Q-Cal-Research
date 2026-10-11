"""Invariants of the calibrators and the two-threshold procedure (cycle 2026-10, PR-E).

These hold for any input, so Hypothesis searches for a counterexample. The scores are drawn
from a coarse grid so exact ties (where pool-adjacent-violators and duplicate merging live)
are common. Each property says which part of arXiv:2405.20459 or of the estimator's
definition it encodes. Reference implementations: ``tests/reference/calibration.py``.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st

from qcal.protocols import ImageDetections
from qcal_lab.calib.base import CalibrationError, IdentityCalibrator
from qcal_lab.calib.isotonic import IsotonicRegression, merge_duplicates
from qcal_lab.calib.platt import PlattScaling
from qcal_lab.calib.thresholds import select_threshold, threshold_grid
from qcal_lab.calib.two_threshold import (
    SplitData,
    ThresholdedCalibration,
    keep_class,
    select_class_thresholds,
    train_calibration,
)
from qcal_lab.data.coco import GroundTruth
from tests.lab_support import det, ground_truth, image, lab_config
from tests.reference.calibration import (
    apply_reference,
    interpolate,
    isotonic_reference,
    logit,
    platt_gradient,
    platt_loss,
    select_reference,
)

pytestmark = pytest.mark.rule("C3")  # calibrators fit on one split; thresholds on another

GRID = st.integers(0, 40).map(lambda k: k / 40)
INNER_GRID = st.integers(1, 39).map(lambda k: k / 40)
TARGET = st.floats(0.0, 1.0, allow_nan=False)
PAIRS = st.lists(st.tuples(GRID, TARGET), min_size=1, max_size=40)


def isotonic(**overrides: object) -> IsotonicRegression:
    settings: dict[str, object] = {"y_min": 0.0, "y_max": 1.0, "out_of_bounds": "clip"}
    settings.update(overrides)
    return IsotonicRegression(**settings)  # type: ignore[arg-type]


def platt() -> PlattScaling:
    return PlattScaling(
        epsilon=1e-7,
        max_iterations=100,
        gradient_tolerance=1e-9,
        ridge=1e-12,
        armijo=1e-4,
        min_step=1e-12,
    )


def fitted_isotonic(pairs: Sequence[tuple[float, float]]) -> IsotonicRegression:
    model = isotonic()
    model.fit([p for p, _ in pairs], [t for _, t in pairs])
    return model


# -- isotonic regression -------------------------------------------------------------------


@given(PAIRS)
def test_isotonic_matches_the_min_max_reference(pairs: list[tuple[float, float]]) -> None:
    """The fit is the weighted least-squares non-decreasing fit of the per-score means."""
    model = fitted_isotonic(pairs)
    knots_x, knots_y = isotonic_reference([p for p, _ in pairs], [t for _, t in pairs])
    assert model.x == knots_x
    assert model.y == pytest.approx(knots_y, abs=1e-12)


@given(PAIRS, st.lists(st.floats(-0.5, 1.5, allow_nan=False), min_size=2, max_size=30))
def test_isotonic_output_is_non_decreasing_everywhere(
    pairs: list[tuple[float, float]], queries: list[float]
) -> None:
    model = fitted_isotonic(pairs)
    clipped = [min(max(q, 0.0), 1.0) for q in sorted(queries)]
    outputs = model.transform(clipped)
    assert all(a <= b + 1e-15 for a, b in itertools.pairwise(outputs))
    assert all(0.0 <= o <= 1.0 for o in outputs)


@given(st.lists(GRID, min_size=1, max_size=40, unique=True))
def test_isotonic_is_idempotent_on_increasing_targets(scores: list[float]) -> None:
    targets = [s * 0.9 + 0.05 for s in scores]  # strictly increasing in the score
    model = fitted_isotonic(list(zip(scores, targets, strict=True)))
    assert model.y == pytest.approx(sorted(targets), abs=1e-15)


@given(PAIRS, st.randoms(use_true_random=False))
def test_isotonic_knots_do_not_depend_on_input_order(
    pairs: list[tuple[float, float]], rng: random.Random
) -> None:
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    a, b = fitted_isotonic(pairs), fitted_isotonic(shuffled)
    assert (a.x, a.y) == (b.x, b.y)  # merging sorts by (score, target): bit-identical


@given(PAIRS, st.integers(2, 4))
def test_isotonic_is_invariant_to_repeating_every_sample(
    pairs: list[tuple[float, float]], times: int
) -> None:
    a, b = fitted_isotonic(pairs), fitted_isotonic(pairs * times)
    assert a.x == b.x
    assert a.y == pytest.approx(b.y, abs=1e-12)


@given(PAIRS)
def test_isotonic_values_survive_a_strictly_increasing_score_remap(
    pairs: list[tuple[float, float]],
) -> None:
    remapped = [(p * p, t) for p, t in pairs]  # strictly increasing on [0, 1]
    assert fitted_isotonic(pairs).y == fitted_isotonic(remapped).y


@given(PAIRS, st.floats(0.0, 0.5))
def test_isotonic_shifting_every_target_shifts_the_fit(
    pairs: list[tuple[float, float]], shift: float
) -> None:
    low = [(p, t * 0.5) for p, t in pairs]  # targets in [0, 0.5]: no clipping after the shift
    high = [(p, t + shift) for p, t in low]
    assert fitted_isotonic(high).y == pytest.approx(
        [v + shift for v in fitted_isotonic(low).y], abs=1e-12
    )


@given(PAIRS, st.floats(-0.5, 1.5, allow_nan=False))
def test_isotonic_interpolates_between_knots_and_clips_outside(
    pairs: list[tuple[float, float]], query: float
) -> None:
    model = fitted_isotonic(pairs)
    assert model.transform([min(max(query, 0.0), 1.0)])[0] == pytest.approx(
        interpolate(model.x, model.y, min(max(query, 0.0), 1.0)), abs=1e-12
    )
    for x, y in zip(model.x, model.y, strict=True):
        assert model.transform([x]) == [y]  # exact at every knot


def test_isotonic_raise_mode_refuses_scores_outside_the_fitted_range() -> None:
    model = isotonic(out_of_bounds="raise")
    model.fit([0.2, 0.4, 0.6], [0.1, 0.3, 0.5])
    assert model.transform([0.2, 0.6]) == [0.1, 0.5]
    with pytest.raises(CalibrationError, match="outside the fitted range"):
        model.transform([0.1])


def test_isotonic_one_distinct_score_fits_the_mean() -> None:
    model = fitted_isotonic([(0.5, 0.2), (0.5, 0.6), (0.5, 0.7)])
    assert model.x == [0.5]
    assert model.y == pytest.approx([0.5])
    assert model.transform([0.0, 1.0]) == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    ("gap", "resolution", "merged"),
    [
        (0.9e-15, 1e-15, True),  # below float64 resolution: one score (scikit-learn float64)
        (2e-15, 1e-15, False),
        (5e-7, 1e-6, True),  # the float32 setting merges what float64 keeps apart
        (2e-6, 1e-6, False),
    ],
)
def test_duplicate_resolution_boundary(gap: float, resolution: float, merged: bool) -> None:
    ux, _, counts = merge_duplicates([0.5, 0.5 + gap], [0.1, 0.9], resolution)
    assert (len(ux) == 1) is merged
    assert sum(counts) == 2


def test_isotonic_clips_to_its_bounds() -> None:
    model = isotonic(y_min=0.2, y_max=0.7)
    model.fit([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])
    assert model.y == [0.2, 0.5, 0.7]


# -- Platt scaling (Eq. 8 and 9: cross-entropy against soft IoU targets, a >= 0) ------------

EPSILON = 1e-7
SCORES = st.lists(INNER_GRID, min_size=2, max_size=40)


def platt_inputs(
    scores: list[float], targets: list[float]
) -> tuple[list[float], list[float], PlattScaling]:
    model = platt()
    model.fit(scores, targets)
    return [logit(p, EPSILON) for p in scores], targets, model


@given(SCORES, st.data())
def test_platt_returns_a_constrained_optimum(scores: list[float], data: st.DataObject) -> None:
    """Interior: zero gradient. Boundary a = 0: zero b-gradient, and a cannot decrease the loss."""
    assume(len(set(scores)) >= 2)
    targets = data.draw(st.lists(st.floats(0.05, 0.95), min_size=len(scores), max_size=len(scores)))
    x, y, model = platt_inputs(scores, targets)
    ga, gb = platt_gradient(x, y, model.a, model.b)
    if model.a > 0.0:
        assert model.converged
        assert max(abs(ga), abs(gb)) < 1e-6
    else:
        assert model.a == 0.0
        assert abs(gb) < 1e-6
        assert ga >= -1e-6  # moving into a > 0 would not lower the loss


@given(SCORES, st.data())
def test_platt_never_loses_to_the_identity_or_the_constant(
    scores: list[float], data: st.DataObject
) -> None:
    targets = data.draw(st.lists(TARGET, min_size=len(scores), max_size=len(scores)))
    x, y, model = platt_inputs(scores, targets)
    mean = min(max(sum(y) / len(y), EPSILON), 1 - EPSILON)
    baseline = min(platt_loss(x, y, 1.0, 0.0), platt_loss(x, y, 0.0, logit(mean, EPSILON)))
    assert platt_loss(x, y, model.a, model.b) <= baseline + 1e-9


@given(SCORES, st.data(), st.randoms(use_true_random=False))
def test_platt_does_not_depend_on_sample_order(
    scores: list[float], data: st.DataObject, rng: random.Random
) -> None:
    assume(len(set(scores)) >= 2)  # with one distinct score (a, b) is not identifiable
    targets = data.draw(st.lists(st.floats(0.05, 0.95), min_size=len(scores), max_size=len(scores)))
    pairs = list(zip(scores, targets, strict=True))
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    a = platt_inputs(scores, targets)[2]
    b = platt_inputs([p for p, _ in shuffled], [t for _, t in shuffled])[2]
    assert (a.a, a.b) == pytest.approx((b.a, b.b), abs=1e-7)


@given(SCORES, st.data(), st.integers(2, 3))
def test_platt_is_invariant_to_repeating_every_sample(
    scores: list[float], data: st.DataObject, times: int
) -> None:
    assume(len(set(scores)) >= 2)  # with one distinct score (a, b) is not identifiable
    targets = data.draw(st.lists(st.floats(0.05, 0.95), min_size=len(scores), max_size=len(scores)))
    a = platt_inputs(scores, targets)[2]
    b = platt_inputs(scores * times, targets * times)[2]
    assert (a.a, a.b) == pytest.approx((b.a, b.b), abs=1e-6)


@given(SCORES, st.data())
def test_platt_flipping_scores_and_targets_negates_the_bias(
    scores: list[float], data: st.DataObject
) -> None:
    """sigmoid(a * logit(1 - p) - b) = 1 - sigmoid(a * logit(p) + b)."""
    assume(len(set(scores)) >= 2)  # with one distinct score (a, b) is not identifiable
    targets = data.draw(st.lists(st.floats(0.05, 0.95), min_size=len(scores), max_size=len(scores)))
    a = platt_inputs(scores, targets)[2]
    b = platt_inputs([1 - p for p in scores], [1 - t for t in targets])[2]
    assert (b.a, b.b) == pytest.approx((a.a, -a.b), abs=1e-6)


@given(
    st.lists(INNER_GRID, min_size=3, max_size=40).filter(lambda s: len(set(s)) >= 2),
    st.floats(0.2, 3.0),
    st.floats(-2.0, 2.0),
)
@example(scores=[0.25, 0.5, 0.75], a_true=1.0, b_true=0.0)
def test_platt_recovers_a_known_calibration_map(
    scores: list[float], a_true: float, b_true: float
) -> None:
    """With targets exactly sigmoid(a* logit(p) + b*), the minimiser is (a*, b*)."""
    targets = [1 / (1 + math.exp(-(a_true * logit(p, EPSILON) + b_true))) for p in scores]
    model = platt_inputs(scores, targets)[2]
    assert (model.a, model.b) == pytest.approx((a_true, b_true), abs=1e-5)


@given(INNER_GRID, st.lists(st.floats(0.05, 0.95), min_size=1, max_size=20))
def test_platt_with_one_distinct_score_predicts_the_mean_target(
    score: float, targets: list[float]
) -> None:
    """Only a * logit(p) + b is identifiable; the prediction at p must be the mean target."""
    model = platt_inputs([score] * len(targets), targets)[2]
    assert model.transform([score])[0] == pytest.approx(sum(targets) / len(targets), abs=1e-6)


@given(st.lists(INNER_GRID, min_size=2, max_size=40, unique=True))
def test_platt_anti_correlated_targets_give_the_boundary_solution(scores: list[float]) -> None:
    """Targets decreasing in the score: the constrained optimum has a = 0, sigmoid(b) = mean."""
    targets = [1.0 - p for p in scores]
    model = platt_inputs(scores, targets)[2]
    assert model.a == 0.0
    assert 1 / (1 + math.exp(-model.b)) == pytest.approx(sum(targets) / len(targets), abs=1e-9)


@given(SCORES, st.data())
def test_platt_is_bitwise_deterministic(scores: list[float], data: st.DataObject) -> None:
    targets = data.draw(st.lists(TARGET, min_size=len(scores), max_size=len(scores)))
    a, b = platt_inputs(scores, targets)[2], platt_inputs(scores, targets)[2]
    assert (a.a, a.b, a.iterations) == (b.a, b.b, b.iterations)


@given(SCORES, st.data())
def test_platt_output_is_monotone(scores: list[float], data: st.DataObject) -> None:
    targets = data.draw(st.lists(TARGET, min_size=len(scores), max_size=len(scores)))
    model = platt_inputs(scores, targets)[2]
    grid = [k / 100 for k in range(101)]
    outputs = model.transform(grid)
    assert all(a <= b for a, b in itertools.pairwise(outputs))


def test_platt_separable_targets_stay_finite_and_ordered() -> None:
    model = platt()
    model.fit([0.1, 0.2, 0.8, 0.9], [0.0, 0.0, 1.0, 1.0])
    outputs = model.transform([0.1, 0.5, 0.9])
    assert all(math.isfinite(o) for o in outputs)
    assert outputs[0] < outputs[1] < outputs[2]


def test_platt_all_zero_targets_push_every_score_towards_zero() -> None:
    model = platt()
    model.fit([0.2, 0.5, 0.8], [0.0, 0.0, 0.0])
    assert max(model.transform([0.2, 0.5, 0.8])) < 0.01


def test_platt_handles_scores_of_exactly_zero_and_one() -> None:
    model = platt()
    model.fit([0.0, 0.5, 1.0], [0.1, 0.5, 0.9])
    assert all(0.0 < o < 1.0 for o in model.transform([0.0, 1.0]))


# -- thresholds and the two-threshold procedure (Alg. A.1/A.2) -----------------------------


@given(
    st.lists(
        st.one_of(st.floats(0, 10), st.just(math.nan), st.just(math.inf)), min_size=1, max_size=20
    )
)
def test_select_threshold_matches_the_reference(values: list[float]) -> None:
    candidates = [i / len(values) for i in range(len(values))]
    by_candidate = dict(zip(candidates, values, strict=True))
    assert select_threshold(candidates, by_candidate.__getitem__) == select_reference(
        candidates, by_candidate.__getitem__
    )


@given(st.floats(0.0, 0.5), st.floats(0.5, 1.0), st.floats(0.01, 0.5))
# Found by Hypothesis 6.168.5 in CI: the last point drifted to 1.000000000024, past stop and 1.0.
@example(start=2.3728313513401887e-11, stop=1.0, step=0.5)
def test_threshold_grid_points_lie_in_range_and_ascend(
    start: float, stop: float, step: float
) -> None:
    grid = threshold_grid(start, stop, step)
    assert grid[0] == round(start, 12)
    assert all(start - 1e-12 <= g <= stop + 1e-12 for g in grid)
    assert list(grid) == sorted(set(grid))


DETECTIONS = st.lists(st.tuples(GRID, st.integers(0, 2)), max_size=12)
IMAGES = st.lists(DETECTIONS, min_size=1, max_size=5)


def images_from(raw: list[list[tuple[float, int]]]) -> tuple[ImageDetections, ...]:
    return tuple(
        image(str(i), *(det(score, label) for score, label in dets)) for i, dets in enumerate(raw)
    )


@given(IMAGES, st.integers(0, 2), GRID, GRID)
def test_keep_class_keeps_every_image_and_nests(
    raw: list[list[tuple[float, int]]], label: int, low: float, high: float
) -> None:
    low, high = min(low, high), max(low, high)
    images = images_from(raw)
    loose, tight = keep_class(images, label, low), keep_class(images, label, high)
    assert [i.image_id for i in loose] == [i.image_id for i in images]
    for kept, original in zip(loose, images, strict=True):
        expected = [d for d in original.detections if d.label == label and d.score >= low]
        assert list(kept.detections) == expected
    for small, big in zip(tight, loose, strict=True):
        assert set(small.detections) <= set(big.detections)


class LowestScoreLoop:
    """A transparent objective: prefer the threshold nearest a per-class target score."""

    def __init__(self, target: float = 0.5) -> None:
        self.target = target
        self.metric_calls = 0

    def targets(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> dict[str, list[float]]:
        return {i.image_id: [d.score * 0.8 for d in i.detections] for i in predictions}

    def threshold_objective(
        self,
        predictions: Sequence[ImageDetections],
        ground_truth: GroundTruth,
        *,
        label: int,
        stage: str,
    ) -> float:
        kept = [d.score for i in predictions for d in i.detections]
        return abs(min(kept) - self.target) + label * 0.0 if kept else math.nan

    def metrics(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> dict[str, float]:
        self.metric_calls += 1
        return {"n": float(sum(len(i.detections) for i in predictions))}


GT = ground_truth(images=5, categories=(1, 2, 3))
CANDIDATES = (0.0, 0.25, 0.5, 0.75)


def train(
    fit: tuple[ImageDetections, ...],
    select: tuple[ImageDetections, ...],
    tmp_path: Path,
    scope: str = "per_class",
) -> ThresholdedCalibration:
    lab = lab_config(tmp_path)
    from qcal_lab.calib import build_calibrator

    return train_calibration(
        fit=SplitData("calibrator_fit_split", fit, GT),
        calibration_threshold_data=SplitData("val", select, GT),
        operating_threshold_data=SplitData("val", select, GT),
        build=lambda: build_calibrator("isotonic", lab),
        scope=scope,
        candidates=CANDIDATES,
        eval_loop=LowestScoreLoop(),
    )


@given(IMAGES, IMAGES)
def test_thresholds_come_from_the_grid_for_every_class(
    fit: list[list[tuple[float, int]]], select: list[list[tuple[float, int]]]
) -> None:
    calibration = train(images_from(fit), images_from(select), Path())
    for table in (calibration.calibration_thresholds, calibration.operating_thresholds):
        assert set(table) == {0, 1, 2}
        assert set(table.values()) <= set(CANDIDATES)


@given(IMAGES, IMAGES)
def test_apply_is_filter_transform_filter(
    fit: list[list[tuple[float, int]]], select: list[list[tuple[float, int]]]
) -> None:
    calibration = train(images_from(fit), images_from(select), Path())
    images = images_from(select)

    def transform(label: int, score: float) -> float:
        return calibration.calibrator_for(label).transform([score])[0]

    assert list(calibration.apply(images)) == apply_reference(
        images,
        calibration.calibration_thresholds,
        calibration.operating_thresholds,
        transform,
    )


@given(IMAGES)
def test_classes_without_fit_detections_get_the_identity(
    select: list[list[tuple[float, int]]],
) -> None:
    fit = images_from([[(0.9, 0), (0.6, 0)]])  # only class 0 has detections to fit on
    calibration = train(fit, images_from(select), Path())
    assert isinstance(calibration.calibrator_for(1), IdentityCalibrator)
    assert isinstance(calibration.calibrator_for(2), IdentityCalibrator)


@given(IMAGES, IMAGES, st.randoms(use_true_random=False))
def test_thresholds_do_not_depend_on_image_order(
    fit: list[list[tuple[float, int]]],
    select: list[list[tuple[float, int]]],
    rng: random.Random,
) -> None:
    shuffled = list(images_from(select))
    rng.shuffle(shuffled)
    a = train(images_from(fit), images_from(select), Path())
    b = train(images_from(fit), tuple(shuffled), Path())
    assert a.calibration_thresholds == b.calibration_thresholds
    assert a.operating_thresholds == b.operating_thresholds


def test_select_class_thresholds_reports_fallbacks() -> None:
    data = SplitData("val", images_from([[(0.9, 0)]]), GT)
    chosen, fallbacks = select_class_thresholds(
        data, [0, 1], CANDIDATES, LowestScoreLoop(), "calibration"
    )
    assert chosen == {0: 0.0, 1: 0.0}  # class 1 has nothing: falls back to the first candidate
    assert fallbacks == [1]
