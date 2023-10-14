"""Tests for igts module."""

import numpy as np
import pytest

from sktime.annotation.igts import IGTS, InformationGainSegmentation, entropy


@pytest.fixture
def multivariate_mean_shift():
    """Generate simple mean shift time series."""
    x = np.concatenate(tuple(np.ones(5) * i**2 for i in range(4)))
    return np.vstack((x, x.max() - x)).T


def test_entropy():
    """Test entropy function."""
    assert entropy(np.ones((10, 1))) == 0.0


def test_igts_identity():
    """Test identity segmentation."""
    X = np.random.random(12).reshape(4, 3)
    id_change_points = IGTS().identity(X)
    assert id_change_points == [0, 4]


def test_igts_get_candidates():
    """Test get_candidates function."""
    candidates = IGTS(step=2).get_candidates(n_samples=10, change_points=[0, 4, 6])
    assert candidates == [2, 8]


def test_IGTS_find_change_points(multivariate_mean_shift):
    """Test the IGTS core estimator."""
    igts = IGTS(k_max=3, step=1)
    pred = igts.find_change_points(multivariate_mean_shift)
    assert isinstance(pred, list)
    assert len(pred) == 5


def test_InformationGainSegmentation(multivariate_mean_shift):
    """Test the InformationGainSegmentation."""
    igts = InformationGainSegmentation(k_max=3, step=1)
    assert igts.get_params() == {"k_max": 3, "step": 1}
    pred = igts.fit_predict(multivariate_mean_shift)
    assert np.array_equal(
        pred,
        np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=np.int32
        ),
    )
    assert igts.change_points_ == [0, 5, 10, 15, 20]
    assert len(igts.intermediate_results_) == 3
