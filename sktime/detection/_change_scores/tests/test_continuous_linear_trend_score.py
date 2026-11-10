"""Tests for ContinuousLinearTrendScore."""

import numpy as np
import pytest

from sktime.detection._change_scores import ContinuousLinearTrendScore
from sktime.detection._change_scores._continuous_linear_trend_score import (
    _continuous_piecewise_linear_trend_contrast_weights,
    _continuous_piecewise_linear_trend_squared_contrast,
)
from sktime.tests.test_switch import run_test_module_changed


def _reference_squared_contrast(signal, start, split, end):
    """Squared contrast computed element by element, as in the original loop."""
    s = start - 1
    e = end - 1
    b = split

    length = e - s
    alpha = np.sqrt(
        6.0
        / (
            length
            * (length**2 - 1)
            * (1 + (e - b + 1) * (b - s) + (e - b) * (b - s - 1))
        )
    )
    beta = np.sqrt(((e - b + 1.0) * (e - b)) / ((b - s - 1.0) * (b - s)))

    first_slope = 3.0 * (b - s) + (e - b) - 1.0
    first_constant = b * (e - s - 1.0) + 2.0 * (s + 1.0) * (b - s)
    second_slope = 3.0 * (e - b) + (b - s) + 1.0
    second_constant = b * (e - s - 1.0) + 2.0 * e * (e - b + 1)

    contrast = 0.0
    for t in range(s + 1, b + 1):
        contrast += (alpha * beta * (first_slope * t - first_constant)) * signal[
            t - start
        ]
    for t in range(b + 1, e + 1):
        contrast += ((-alpha / beta) * (second_slope * t - second_constant)) * signal[
            t - start
        ]
    return contrast**2


def _random_cuts(n, n_intervals, max_length, rng):
    starts = rng.integers(0, n - 10, n_intervals)
    ends = np.minimum(starts + rng.integers(6, max_length, n_intervals), n)
    splits = starts + 2 + rng.integers(0, ends - starts - 4)
    return np.column_stack((starts, splits, ends))


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_squared_contrast_matches_reference():
    """Vectorized squared contrast matches the element-wise reference."""
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.standard_normal(100))
    for start, split, end in _random_cuts(100, 50, 60, rng):
        actual = _continuous_piecewise_linear_trend_squared_contrast(
            X[start:end], start, split, end
        )
        expected = _reference_squared_contrast(X[start:end], start, split, end)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_contrast_weights_are_shift_invariant_and_orthogonal_to_trends():
    """Weights depend only on length and split offset, and annihilate lines."""
    weights = _continuous_piecewise_linear_trend_contrast_weights(10, 14, 30)
    weights_shifted = _continuous_piecewise_linear_trend_contrast_weights(0, 4, 20)
    np.testing.assert_allclose(weights, weights_shifted)

    # a linear trend has zero contrast
    t = np.arange(20.0)
    np.testing.assert_allclose(weights @ (3.0 + 2.0 * t), 0.0, atol=1e-10)
    # the weights are normalised
    np.testing.assert_allclose(weights @ weights, 1.0)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("n_columns", [1, 3])
def test_analytical_score_random_intervals(n_columns):
    """Score on random intervals matches the reference, per interval and column."""
    rng = np.random.default_rng(1)
    n = 300
    X = np.cumsum(rng.standard_normal((n, n_columns)), axis=0)
    cuts = _random_cuts(n, 100, 80, rng)

    scores = ContinuousLinearTrendScore().evaluate(X, cuts)

    expected = np.zeros((len(cuts), n_columns))
    for i, (start, split, end) in enumerate(cuts):
        for j in range(n_columns):
            expected[i, j] = _reference_squared_contrast(
                X[start:end, j], start, split, end
            )
    np.testing.assert_allclose(scores, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_analytical_score_grouped_intervals():
    """Intervals with equal length and split offset are evaluated at once."""
    rng = np.random.default_rng(2)
    n = 200
    X = np.cumsum(rng.standard_normal((n, 2)), axis=0)
    # all splits of several intervals of the same length
    cuts = np.array(
        [[s, s + k, s + 30] for s in [0, 10, 50, 120, 170] for k in range(2, 28)]
    )

    scores = ContinuousLinearTrendScore().evaluate(X, cuts)

    expected = np.vstack(
        [ContinuousLinearTrendScore().evaluate(X, cut[None, :]) for cut in cuts]
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_analytical_score_agrees_with_regression():
    """The analytical score agrees with the linear regression score."""
    rng = np.random.default_rng(3)
    n = 200
    X = np.cumsum(rng.standard_normal((n, 2)), axis=0)
    cuts = _random_cuts(n, 50, 100, rng)

    analytical = ContinuousLinearTrendScore().evaluate(X, cuts)
    X_with_time = np.column_stack((np.arange(n, dtype=float), X))
    regression = ContinuousLinearTrendScore(time_column=0).evaluate(X_with_time, cuts)

    np.testing.assert_allclose(analytical, regression, rtol=1e-8, atol=1e-8)
