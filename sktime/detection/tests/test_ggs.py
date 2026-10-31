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
    not run_test_for_class(GreedyGaussianSegmentation),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_GreedyGaussianSegmentation_output_format(univariate_mean_shift):
    """predict returns segments as intervals covering the series, see #7462."""
    import pandas as pd

    X = pd.DataFrame(univariate_mean_shift)
    ggs = GreedyGaussianSegmentation(k_max=3, lamb=1.0, random_state=0)
    y = ggs.fit_predict(X)

    assert list(y.columns) == ["ilocs"]
    intervals = list(y["ilocs"])
    assert all(isinstance(iv, pd.Interval) for iv in intervals)
    assert intervals[0].left == 0
    assert intervals[-1].right == len(X)
    for prev, nxt in zip(intervals[:-1], intervals[1:]):
        assert prev.right == nxt.left
    # the four mean levels are found
    assert [iv.left for iv in intervals] == [0, 5, 10, 15]

    # dense labels via transform, and change points via predict_points
    labels = ggs.transform(X).iloc[:, 0].to_numpy()
    np.testing.assert_array_equal(labels, np.repeat([0, 1, 2, 3], 5))
    assert list(ggs.predict_points(X)["ilocs"]) == [0, 5, 10, 15]
