# -*- coding: utf-8 -*-
"""Test suite for numba pairwise distances."""

__author__ = ["chrisholder"]

from typing import Callable, List

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from sktime.dists_kernels.numba.distances._registry import NUMBA_DISTANCES
from sktime.dists_kernels.tests._utils import create_test_distance_numpy

pairwises_and_distances = [[dist[2], dist[1]] for dist in NUMBA_DISTANCES]


def validate_result(pairwise_result: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Validate a pairwise result.

    Parameters
    ----------
    pairwise_result: np.ndarray (2D array)
        [n, m] matrix with pairwise results where n is len(x) and m is len(y).
    x: np.ndarray
        First timeseries used to generate pairwise.
    y: np.ndarray
        Second timeseries used to generate pairwise.
    """
    assert isinstance(pairwise_result, np.ndarray)
    assert pairwise_result.ndim == 2
    assert pairwise_result.shape[0] is len(x)
    assert pairwise_result.shape[1] is len(y)


@pytest.mark.parametrize("pairwise_dists", pairwises_and_distances)
def test_series_pairwise_distances(pairwise_dists: List[Callable]):
    """Test pairwise distances with series formatted timeseries.

    Parameters
    ----------
    pairwise_dists: List[Callable]
        List containing the pairwise function as first value and distance as second.
    """
    pairwise = pairwise_dists[0]
    distance = pairwise_dists[1]
    if pairwise is None:
        return

    # Test equal length timeseries
    # Test univariate distance where x is 10x1 numpy array and y is 15x1 numpy array
    x_univariate = create_test_distance_numpy(10, 1)
    y_univariate = create_test_distance_numpy(15, 1, random_state=2)
    univariate_pairwise_result = pairwise(x_univariate, y_univariate)
    univariate_distance_result = distance(x_univariate, y_univariate)
    validate_result(univariate_pairwise_result, x_univariate, y_univariate)
    # The diagonal of the univariate pairwise should equal distance call
    assert_almost_equal(univariate_distance_result, univariate_pairwise_result.trace())

    # Test multivariate where x is 10x10 numpy array and y is 15x10 numpy array
    x_multivariate = create_test_distance_numpy(10, 10)
    y_multivariate = create_test_distance_numpy(15, 10, random_state=2)
    multivariate_pairwise_result = pairwise(x_multivariate, y_multivariate)
    multivariate_result = distance(x_multivariate, y_multivariate)
    validate_result(multivariate_pairwise_result, x_multivariate, y_multivariate)
    # The diagonal of the multivariate pairwise should equal distance call
    assert_almost_equal(multivariate_result, multivariate_pairwise_result.trace())

    # Test univariate where x is size 10 numpy array and y is size 15 numpy array
    single_ts_x = create_test_distance_numpy(10)
    single_ts_y = create_test_distance_numpy(15, random_state=2)
    single_pairwise_result = pairwise(single_ts_x, single_ts_y)
    single_result = distance(single_ts_x, single_ts_y)
    validate_result(single_pairwise_result, single_ts_x, single_ts_y)
    # The diagonal of single pairwise should equal the distance call
    assert_almost_equal(single_result, single_pairwise_result.trace())


@pytest.mark.parametrize("pairwise_dists", pairwises_and_distances)
def test_panel_pairwise_distances(pairwise_dists: List[Callable]):
    """Test pairwise distances with series formatted timeseries.

    Parameters
    ----------
    pairwise_dists: List[Callable]
        List containing the pairwise function as first value and distance as second.
    """
    pairwise = pairwise_dists[0]
    distance = pairwise_dists[1]
    if pairwise is None:
        return

    # Test univariate panel where x is 10x1x10 and y is 15x1x10
    x_univariate = create_test_distance_numpy(10, 1, 10)
    y_univariate = create_test_distance_numpy(15, 1, 10, random_state=2)
    univariate_pairwise_result = pairwise(x_univariate, y_univariate)
    univariate_distance_result = distance(x_univariate, y_univariate)
    validate_result(univariate_pairwise_result, x_univariate, y_univariate)
    # The diagonal of the univariate pairwise should equal distance call
    assert_almost_equal(univariate_distance_result, univariate_pairwise_result.trace())

    # Test multivariate panel where x is 10x10x10 and y is 15x10x10
    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(15, 10, 10, random_state=2)
    multivariate_pairwise_result = pairwise(x_multivariate, y_multivariate)
    multivariate_result = distance(x_multivariate, y_multivariate)
    validate_result(multivariate_pairwise_result, x_multivariate, y_multivariate)
    # The diagonal of the multivariate pairwise should equal distance call
    assert_almost_equal(multivariate_result, multivariate_pairwise_result.trace())
