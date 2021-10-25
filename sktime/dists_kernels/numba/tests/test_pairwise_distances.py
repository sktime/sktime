# -*- coding: utf-8 -*-
"""Test suite for numba pairwise distances."""

__author__ = ["chrisholder"]

from typing import Callable

import numpy as np
import pytest

from sktime.dists_kernels.numba.distances._registry import NUMBA_DISTANCES
from sktime.dists_kernels.tests._utils import create_test_distance_numpy

pairwises = [dist[2] for dist in NUMBA_DISTANCES]


def validate_result(pairwise_result: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Validate a pairwise result.

    Parameters
    ----------
    pairwise_result: np.ndarray (2D array)
        [n, m] matrix with pairwise results
    x: np.ndarray
        First timeseries used to generate pairwise.
    y: np.ndarray
        Second timeseries used to generate pairwise
    """
    # Expect a numpy array of size nxm where n is len(x) and m is len(m)
    assert isinstance(pairwise_result, np.ndarray)
    assert pairwise_result.ndim == 2
    assert pairwise_result.shape[0] is len(x)
    assert pairwise_result.shape[1] is len(y)


@pytest.mark.parametrize("pairwise", pairwises)
def test_series_pairwise_distances(pairwise: Callable):
    """Test pairwise distances with series formatted timeseries.

    Parameters
    ----------
    pairwise: Callable
        Pairwise function to run test on.
    """
    if pairwise is None:
        return

    # Test equal length timeseries
    # Test univariate distance where x is 10x1 numpy array and y is 15x1 numpy array
    x_univariate = create_test_distance_numpy(10, 1)
    y_univariate = create_test_distance_numpy(15, 1, random_state=2)
    univariate_pairwise_result = pairwise(x_univariate, y_univariate)
    validate_result(univariate_pairwise_result, x_univariate, y_univariate)

    # Test multivariate where x is 10x10 numpy array and y is 15x10 numpy array
    x_multivariate = create_test_distance_numpy(10, 10)
    y_multivariate = create_test_distance_numpy(15, 10, random_state=2)
    multivariate_pairwise_result = pairwise(x_multivariate, y_multivariate)
    validate_result(multivariate_pairwise_result, x_multivariate, y_multivariate)

    # Test univariate where x is size 10 numpy array and y is size 15 numpy array
    single_ts_x = create_test_distance_numpy(10)
    single_ts_y = create_test_distance_numpy(15, random_state=2)
    single_result = pairwise(single_ts_x, single_ts_y)
    validate_result(single_result, single_ts_x, single_ts_y)


@pytest.mark.parametrize("pairwise", pairwises)
def test_panel_pairwise_distances(pairwise: Callable):
    """Test pairwise with panel formatted timeseries.

    Parameters
    ----------
    pairwise: Callable
        Pairwise function to run test on.
    """
    if pairwise is None:
        return

    # Test univariate panel where x is 10x1x10 and y is 15x1x10
    x_univariate = create_test_distance_numpy(10, 1, 10)
    y_univariate = create_test_distance_numpy(15, 1, 10, random_state=2)
    univariate_pairwise_result = pairwise(x_univariate, y_univariate)
    validate_result(univariate_pairwise_result, x_univariate, y_univariate)

    # Test multivariate panel where x is 10x10x10 and y is 15x10x10
    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(15, 10, 10, random_state=2)
    multivariate_pairwise_result = pairwise(x_multivariate, y_multivariate)
    validate_result(multivariate_pairwise_result, x_multivariate, y_multivariate)
