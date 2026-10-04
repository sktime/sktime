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


@pytest.mark.parametrize("random_state", [None, 42])
def test_ggs_does_not_modify_global_numpy_rng(univariate_mean_shift, random_state):
    """Test that GGS does not modify the process-global NumPy RNG state."""
    ggs = GreedyGaussianSegmentation(k_max=5, lamb=0.5, random_state=random_state)

    np.random.seed(42)
    state_before = np.random.get_state()
    ggs.fit_predict(univariate_mean_shift)
    state_after = np.random.get_state()

    assert all(
        np.array_equal(before, after)
        if isinstance(before, np.ndarray)
        else before == after
        for before, after in zip(state_before, state_after)
    )


def test_ggs_explicit_random_state_is_reproducible(univariate_mean_shift):
    """Test that equal explicit random states produce equal segmentations."""
    prediction_1 = GreedyGaussianSegmentation(
        k_max=5, lamb=0.5, random_state=42
    ).fit_predict(univariate_mean_shift)
    prediction_2 = GreedyGaussianSegmentation(
        k_max=5, lamb=0.5, random_state=42
    ).fit_predict(univariate_mean_shift)

    np.testing.assert_array_equal(prediction_1, prediction_2)
