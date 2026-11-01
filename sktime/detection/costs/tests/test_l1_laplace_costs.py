"""Tests for L1Cost and LaplaceCost."""

import numpy as np
import pytest
from scipy.stats import laplace

from sktime.detection._utils import (
    group_intervals_by_length,
    iter_interval_windows,
)
from sktime.detection.costs import L1Cost, LaplaceCost
from sktime.detection.costs._l1_cost import _l1_cost_mle_location
from sktime.detection.costs._laplace_cost import _laplace_cost_mle_params
from sktime.tests.test_switch import run_test_module_changed


def _reference_l1_cost(X, cuts, locations=None):
    costs = np.zeros((len(cuts), X.shape[1]))
    for i, (start, end) in enumerate(cuts):
        for col in range(X.shape[1]):
            x = X[start:end, col]
            loc = np.median(x) if locations is None else locations[col]
            costs[i, col] = np.sum(np.abs(x - loc))
    return costs


def _reference_laplace_cost(X, cuts, params=None):
    costs = np.zeros((len(cuts), X.shape[1]))
    for i, (start, end) in enumerate(cuts):
        for col in range(X.shape[1]):
            x = X[start:end, col]
            if params is None:
                loc = np.median(x)
                scale = np.mean(np.abs(x - loc))
            else:
                loc, scale = params[0][col], params[1][col]
            costs[i, col] = -2.0 * np.sum(laplace.logpdf(x, loc=loc, scale=scale))
    return costs


def _make_intervals(n, n_intervals, max_length, rng, min_length=3):
    starts = rng.integers(0, n - min_length, n_intervals)
    ends = np.minimum(starts + rng.integers(min_length, max_length, n_intervals), n)
    return np.column_stack((starts, ends))


# interval sets exercising the per-interval and the grouped evaluation path
INTERVAL_SETS = [(5, 10), (200, 8), (200, 150)]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("n_columns", [1, 3])
@pytest.mark.parametrize("n_intervals, max_length", INTERVAL_SETS)
def test_l1_cost(n_columns, n_intervals, max_length):
    """L1Cost matches the sum of absolute deviations from the median / location."""
    rng = np.random.default_rng(0)
    n = 200
    X = np.cumsum(rng.standard_normal((n, n_columns)), axis=0)
    cuts = _make_intervals(n, n_intervals, max_length, rng)

    costs = L1Cost().evaluate(X, cuts)
    np.testing.assert_allclose(costs, _reference_l1_cost(X, cuts), rtol=1e-10)

    locations = rng.standard_normal(n_columns)
    costs = L1Cost(param=locations).evaluate(X, cuts)
    expected = _reference_l1_cost(X, cuts, locations=locations)
    np.testing.assert_allclose(costs, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("n_columns", [1, 3])
@pytest.mark.parametrize("n_intervals, max_length", INTERVAL_SETS)
def test_laplace_cost(n_columns, n_intervals, max_length):
    """LaplaceCost matches twice the negative scipy Laplace log likelihood."""
    rng = np.random.default_rng(1)
    n = 200
    X = np.cumsum(rng.standard_normal((n, n_columns)), axis=0)
    cuts = _make_intervals(n, n_intervals, max_length, rng)

    costs = LaplaceCost().evaluate(X, cuts)
    np.testing.assert_allclose(costs, _reference_laplace_cost(X, cuts), rtol=1e-10)

    locations = rng.standard_normal(n_columns)
    scales = rng.uniform(0.5, 2.0, n_columns)
    costs = LaplaceCost(param=(locations, scales)).evaluate(X, cuts)
    expected = _reference_laplace_cost(X, cuts, params=(locations, scales))
    np.testing.assert_allclose(costs, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_grouped_and_per_interval_evaluation_agree():
    """Grouped evaluation of equal length intervals agrees with per-interval."""
    rng = np.random.default_rng(2)
    n, n_columns = 300, 2
    X = np.cumsum(rng.standard_normal((n, n_columns)), axis=0)

    # many intervals of few distinct lengths, evaluated grouped
    starts = rng.integers(0, n - 20, 100)
    ends = starts + rng.choice([5, 10, 20], 100)
    groups = group_intervals_by_length(starts, ends)
    assert len(groups) == 3
    assert sorted(np.concatenate(groups)) == list(range(100))

    for func in [_l1_cost_mle_location, _laplace_cost_mle_params]:
        grouped = func(starts, ends, X)
        # same intervals one at a time, evaluated per interval
        per_interval = np.vstack(
            [func(starts[[i]], ends[[i]], X) for i in range(len(starts))]
        )
        np.testing.assert_array_equal(grouped, per_interval)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_iter_interval_windows():
    """iter_interval_windows yields the segments of each interval."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 2))
    starts = np.array([0, 10, 5, 20, 30])
    ends = np.array([5, 15, 15, 25, 33])
    groups = group_intervals_by_length(starts, ends)

    seen = []
    for idx, windows in iter_interval_windows(groups, starts, ends, X):
        for j, i in enumerate(idx):
            np.testing.assert_array_equal(windows[j], X[starts[i] : ends[i]])
            seen.append(i)
    assert sorted(seen) == list(range(len(starts)))


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_docstring_examples():
    """The docstring examples of L1Cost and LaplaceCost."""
    X = np.concatenate([np.zeros(5), 10 * np.ones(5)]).reshape(-1, 1)
    cuts = [[0, 5], [5, 10], [0, 10]]
    np.testing.assert_allclose(L1Cost().evaluate(X, cuts).ravel(), [0.0, 0.0, 50.0])
    np.testing.assert_allclose(
        L1Cost(param=0.0).evaluate(X, cuts).ravel(), [0.0, 50.0, 50.0]
    )

    X = np.array([0.0, 2.0, 0.0, 2.0, 10.0, 12.0, 10.0, 12.0]).reshape(-1, 1)
    cuts = [[0, 4], [4, 8], [0, 8]]
    np.testing.assert_allclose(
        LaplaceCost().evaluate(X, cuts).ravel(), [13.55, 13.55, 52.84], atol=0.01
    )
    np.testing.assert_allclose(
        LaplaceCost(param=(1.0, 1.0)).evaluate(X, cuts).ravel(),
        [13.55, 85.55, 99.09],
        atol=0.01,
    )
