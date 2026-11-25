"""Tests for PoissonCost."""

import numpy as np
import pytest
from scipy.stats import poisson

from sktime.detection.costs import PoissonCost
from sktime.tests.test_switch import run_test_module_changed


def _reference_cost(X, cuts, rates=None):
    """Twice negative Poisson log likelihood via scipy, per interval and column."""
    costs = np.zeros((len(cuts), X.shape[1]))
    for i, (start, end) in enumerate(cuts):
        for col in range(X.shape[1]):
            x = X[start:end, col]
            rate = np.mean(x) if rates is None else rates[col]
            if rate == 0:
                costs[i, col] = 0.0
            else:
                costs[i, col] = -2.0 * np.sum(poisson.logpmf(x, rate))
    return costs


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("n_columns", [1, 3])
@pytest.mark.parametrize("rate", [0.3, 5.0, 50.0])
def test_poisson_cost_optimal_rate(n_columns, rate):
    """Optimal rate cost matches the scipy Poisson log likelihood at the mean."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.poisson(rate, (n, n_columns))
    starts = rng.integers(0, n - 1, 100)
    ends = np.minimum(starts + rng.integers(1, 60, 100), n)
    cuts = np.column_stack((starts, ends))

    costs = PoissonCost().evaluate(X, cuts)
    np.testing.assert_allclose(costs, _reference_cost(X, cuts), rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("n_columns", [1, 2])
def test_poisson_cost_fixed_rate(n_columns):
    """Fixed rate cost matches the scipy Poisson log likelihood at that rate."""
    rng = np.random.default_rng(1)
    n = 300
    X = rng.poisson(4.0, (n, n_columns))
    starts = rng.integers(0, n - 1, 100)
    ends = np.minimum(starts + rng.integers(1, 60, 100), n)
    cuts = np.column_stack((starts, ends))
    rates = np.linspace(1.5, 6.0, n_columns)

    costs = PoissonCost(param=rates if n_columns > 1 else rates[0]).evaluate(X, cuts)
    np.testing.assert_allclose(
        costs, _reference_cost(X, cuts, rates=rates), rtol=1e-9, atol=1e-9
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_poisson_cost_edge_cases():
    """All-zero intervals have zero optimal cost, single samples work."""
    X = np.array([0, 0, 0, 0, 0, 3, 1, 4, 1, 5]).reshape(-1, 1)
    cuts = np.array([[0, 5], [0, 10], [5, 10], [7, 8], [2, 3]])

    costs = PoissonCost().evaluate(X, cuts)
    expected = _reference_cost(X, cuts)
    np.testing.assert_allclose(costs, expected, rtol=1e-9, atol=1e-9)
    assert costs[0, 0] == 0.0
    assert costs[4, 0] == 0.0

    costs = PoissonCost(param=1.5).evaluate(X, cuts)
    expected = _reference_cost(X, cuts, rates=[1.5])
    np.testing.assert_allclose(costs, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_poisson_cost_docstring_example():
    """The docstring example values."""
    X = np.array([1, 2, 1, 2, 20, 22, 20, 22]).reshape(-1, 1)
    cuts = [[0, 4], [4, 8], [0, 8]]
    np.testing.assert_allclose(
        PoissonCost().evaluate(X, cuts).ravel(), [9.91, 19.75, 110.33], atol=0.01
    )
    np.testing.assert_allclose(
        PoissonCost(param=2.0).evaluate(X, cuts).ravel(),
        [10.45, 262.78, 273.23],
        atol=0.01,
    )
