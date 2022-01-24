# -*- coding: utf-8 -*-
"""Test function for MSM."""


import numpy as np
from numpy.testing import assert_almost_equal

from sktime.datasets import load_arrow_head
from sktime.distances import distance_factory
from sktime.distances.tests._utils import create_test_distance_numpy


def test_msm():
    """Test MSM.

    Need to get the equivalent numbers from tsml.
    """
    x_train, y_train = load_arrow_head("train", return_type="numpy3d")
    x_test, y_test = load_arrow_head("test", return_type="numpy3d")
    first = np.transpose(x_train[0])
    second = np.transpose(x_train[1])
    cvalues = [0, 0.01, 0.1, 1, 10]
    cdists = [6.3502, 8.6677, 25.6957, 62.2554, 72.76039]
    for i in range(len(cvalues)):
        msm = distance_factory(metric="msm", c=cvalues[i])
        dist = msm(first, second)
        assert_almost_equal(cdists[i], dist, decimal=3)
    x = create_test_distance_numpy(10, 1, random_state=0)
    y = create_test_distance_numpy(10, 1, random_state=2)
    msm = distance_factory(metric="msm")
    dist = msm(x, y)
    assert_almost_equal(dist, 4.247362, decimal=3)
