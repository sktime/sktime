# -*- coding: utf-8 -*-
"""Tests for GGS module."""

import numpy as np
import pytest

from sktime.annotation.igts import IGTS, InformationGainSegmentation


@pytest.fixture
def univariate_mean_shift():
    """Generate simple mean shift time series."""
    x = np.concatenate(tuple(np.ones(5) * i**2 for i in range(4)))
    return x[:, np.newaxis]


def test_IGTS_find_change_points(univariate_mean_shift):
    """Test the IGTS core estimator."""
    ggs = IGTS(k_max=10, lamb=1.0)
    pred = ggs.find_change_points(univariate_mean_shift)
    assert isinstance(pred, list)
    assert len(pred) == 5


def test_GreedyGaussianSegmentation(univariate_mean_shift):
    """Test the InformationGainSegmentation."""
    igts = InformationGainSegmentation(k_max=5, step=1)
    assert igts.get_params() == {"k_max": 5, "step": 1}
    pred = igts.fit_predict(univariate_mean_shift)
    assert pred == [1, 4]
