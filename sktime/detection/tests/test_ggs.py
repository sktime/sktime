"""Tests for GGS module."""

import numpy as np
import pytest

from sktime.detection.ggs import GGS, GreedyGaussianSegmentation
from sktime.tests.test_switch import run_test_for_class


@pytest.fixture
def univariate_mean_shift():
    """Generate simple mean shift time series."""
    x = np.concatenate(tuple(np.ones(5) * i**2 for i in range(4)))
    return x[:, np.newaxis]


@pytest.mark.skipif(
    not run_test_for_class(GGS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_GGS_find_change_points(univariate_mean_shift):
    """Test the GGS core estimator."""
    ggs = GGS(k_max=10, lamb=1.0)
    pred = ggs.find_change_points(univariate_mean_shift)
    assert isinstance(pred, list)
    assert len(pred) == 5


@pytest.mark.skipif(
    not run_test_for_class(GreedyGaussianSegmentation),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_GreedyGaussianSegmentation(univariate_mean_shift):
    """Test the GreedyGaussianSegmentation."""
    ggs = GreedyGaussianSegmentation(k_max=5, lamb=0.5)
    assert ggs.get_params() == {
        "k_max": 5,
        "lamb": 0.5,
        "verbose": False,
        "max_shuffles": 250,
        "random_state": None,
    }


@pytest.mark.skipif(
    not run_test_for_class(GGS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_series", [1, 2, 4])
@pytest.mark.parametrize("lamb", [0.1, 1.0])
def test_GGS_add_new_change_point_matches_objective(n_series, lamb):
    """Test that the split subroutine minimizes the GGS objective.

    The best split point and its gain in log-likelihood must agree with a direct
    evaluation of ``log_likelihood`` on the two parts, for every candidate split.
    This also covers the multivariate case, where an incorrect initialization of
    the running mean of the left segment used to give a different objective.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(size=(150, n_series))
    data[50:100] += 2.0

    ggs = GGS(lamb=lamb)
    new_index, gain = ggs.add_new_change_point(data)

    candidates = np.arange(2, data.shape[0] - 1)
    objective = np.array(
        [
            ggs.log_likelihood(data[:i]) + ggs.log_likelihood(data[i:])
            for i in candidates
        ]
    )
    expected_index = candidates[int(np.argmin(objective))]
    expected_gain = objective.min() - ggs.log_likelihood(data)

    assert new_index == expected_index
    np.testing.assert_allclose(gain, expected_gain, rtol=1e-8)


@pytest.mark.skipif(
    not run_test_for_class(GGS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_GGS_add_new_change_point_no_improvement():
    """Test that no change point is added if the segment is too short."""
    data = np.ones((3, 2))
    assert GGS(lamb=1.0).add_new_change_point(data) == (0, 0.0)
