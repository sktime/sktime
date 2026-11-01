"""Tests for LinearTrendCost."""

import numpy as np
import pytest

from sktime.detection.costs import LinearTrendCost
from sktime.detection.costs._linear_trend_cost import (
    _group_intervals_by_length,
    _linear_trend_cost_fixed,
    _linear_trend_cost_index_mle,
    _linear_trend_cost_mle,
)
from sktime.tests.test_switch import run_test_module_changed


def _reference_cost(X, cuts, time_steps=None, params=None):
    """Linear trend cost by fitting np.polyfit to each interval and column."""
    costs = np.zeros((len(cuts), X.shape[1]))
    for i, (start, end) in enumerate(cuts):
        if time_steps is None:
            t = np.arange(end - start, dtype=float)
        else:
            t = time_steps[start:end]
        for col in range(X.shape[1]):
            x = X[start:end, col]
            if params is None:
                slope, intercept = np.polyfit(t, x, 1)
            else:
                slope, intercept = params[col]
            costs[i, col] = np.sum((x - (intercept + slope * t)) ** 2)
    return costs


def _make_intervals(n, n_intervals, max_length, rng):
    starts = rng.integers(0, n - 3, n_intervals)
    ends = np.minimum(starts + rng.integers(3, max_length, n_intervals), n)
    return np.column_stack((starts, ends))


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("n_columns", [1, 3])
@pytest.mark.parametrize(
    "n_intervals, max_length",
    [(5, 10), (200, 8), (200, 150)],  # few intervals, repeated lengths, mixed
)
def test_linear_trend_cost_optimal_param(n_columns, n_intervals, max_length):
    """Optimal parameter cost matches a polyfit reference, with and without time."""
    rng = np.random.default_rng(0)
    n = 200
    X = np.cumsum(rng.standard_normal((n, n_columns)), axis=0)
    cuts = _make_intervals(n, n_intervals, max_length, rng)

    costs = LinearTrendCost().evaluate(X, cuts)
    np.testing.assert_allclose(costs, _reference_cost(X, cuts), rtol=1e-8, atol=1e-8)

    time_stamps = np.cumsum(rng.random(n) + 0.5)
    X_with_time = np.column_stack((time_stamps, X))
    costs = LinearTrendCost(time_column=0).evaluate(X_with_time, cuts)
    expected = _reference_cost(X, cuts, time_steps=time_stamps - time_stamps[0])
    np.testing.assert_allclose(costs, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("n_intervals, max_length", [(5, 10), (200, 8), (200, 150)])
def test_linear_trend_cost_fixed_param(n_columns, n_intervals, max_length):
    """Fixed parameter cost matches the sum of squared residuals of the trend."""
    rng = np.random.default_rng(1)
    n = 200
    X = np.cumsum(rng.standard_normal((n, n_columns)), axis=0)
    cuts = _make_intervals(n, n_intervals, max_length, rng)
    params = rng.standard_normal((n_columns, 2))

    # with fixed parameters, time steps are the global index [0, ..., n - 1]
    costs = LinearTrendCost(param=params).evaluate(X, cuts)
    expected = _reference_cost(
        X, cuts, time_steps=np.arange(n, dtype=float), params=params
    )
    np.testing.assert_allclose(costs, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_grouped_and_per_interval_evaluation_agree():
    """Grouped evaluation of equal length intervals agrees with per-interval."""
    rng = np.random.default_rng(2)
    n, n_columns = 300, 2
    X = np.cumsum(rng.standard_normal((n, n_columns)), axis=0)
    time_steps = np.cumsum(rng.random(n) + 0.5)
    params = rng.standard_normal((n_columns, 2))

    # many intervals of few distinct lengths, evaluated grouped
    starts = rng.integers(0, n - 20, 100)
    ends = starts + rng.choice([5, 10, 20], 100)
    groups = _group_intervals_by_length(starts, ends)
    assert len(groups) == 3

    for func, args in [
        (_linear_trend_cost_index_mle, (X,)),
        (_linear_trend_cost_mle, (X, time_steps)),
        (_linear_trend_cost_fixed, (X, time_steps, params)),
    ]:
        grouped = func(starts, ends, *args)
        # same intervals one at a time, evaluated per interval
        per_interval = np.vstack(
            [func(starts[[i]], ends[[i]], *args) for i in range(len(starts))]
        )
        np.testing.assert_allclose(grouped, per_interval, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_linear_trend_cost_exact_lines():
    """Intervals on a straight line have zero cost, as in the docstring example."""
    X = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 9.0, 8.0, 7.0]).reshape(-1, 1)
    cost = LinearTrendCost()
    costs = cost.evaluate(X, [[0, 4], [4, 8], [0, 8]])
    np.testing.assert_allclose(costs[:2, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(costs[2, 0], 33.333333, rtol=1e-6)

    cost = LinearTrendCost(param=np.array([[1.0, 0.0]]), share_fixed_trend=True)
    costs = cost.evaluate(X, [[0, 4], [4, 8], [0, 8]])
    np.testing.assert_allclose(costs[:, 0], [0.0, 56.0, 56.0], atol=1e-12)
