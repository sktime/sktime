# -*- coding: utf-8 -*-
"""Test suite for numba distances."""

__author__ = ["chrisholder"]

from typing import Callable

import pytest

from sktime.dists_kernels.numba.distances._registry import NUMBA_DISTANCES
from sktime.dists_kernels.tests._utils import create_test_distance_numpy

distances = [dist[1] for dist in NUMBA_DISTANCES]


def validate_result(result: float) -> None:
    """Validate a distance result.

    Parameters
    ----------
    result: float
        Result to validate
    """
    assert isinstance(result, float)


@pytest.mark.parametrize("distance", distances)
def test_series_distances(distance: Callable):
    """Test all distances with series timeseries.

    Parameters
    ----------
    distance: Callable
        Distance function to run test on
    """
    # Test univariate distance where x is 10x1 numpy array and y is 15x1 numpy array
    x_univariate = create_test_distance_numpy(10, 1)
    y_univariate = create_test_distance_numpy(15, 1, random_state=2)
    univariate_result = distance(x_univariate, y_univariate)
    validate_result(univariate_result)

    # Test multivariate where x is 10x10 numpy array and y is 15x10 numpy array
    x_multivariate = create_test_distance_numpy(10, 10)
    y_multivariate = create_test_distance_numpy(15, 10, random_state=2)
    multivariate_result = distance(x_multivariate, y_multivariate)
    validate_result(multivariate_result)

    # Test univariate where x is size 10 numpy array and y is size 15 numpy array
    single_ts_x = create_test_distance_numpy(10)
    single_ts_y = create_test_distance_numpy(15, random_state=2)
    single_result = distance(single_ts_x, single_ts_y)
    validate_result(single_result)

    assert single_result == univariate_result


@pytest.mark.parametrize("distance", distances)
def test_panel_distances(distance: Callable):
    """Test all distance with panel timeseries.

    Parameters
    ----------
    distance: Callable
        Distance function to run test on
    """
    # Test univariate panel where x is 10x1x10 and y is 15x1x10
    x_univariate = create_test_distance_numpy(10, 1, 10)
    y_univariate = create_test_distance_numpy(15, 1, 10, random_state=2)
    univariate_result = distance(x_univariate, y_univariate)
    validate_result(univariate_result)

    # Test multivariate panel where x is 10x10x10 and y is 15x10x10
    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(15, 10, 10, random_state=2)
    multivariate_result = distance(x_multivariate, y_multivariate)
    validate_result(multivariate_result)
