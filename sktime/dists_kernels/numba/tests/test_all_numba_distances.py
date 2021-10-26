# -*- coding: utf-8 -*-
"""Test suite for numba distances."""

__author__ = ["chrisholder"]

from typing import Callable, List

import pytest
from numpy.testing import assert_almost_equal

from sktime.dists_kernels.numba.distances._registry import NUMBA_DISTANCES
from sktime.dists_kernels.tests._utils import create_test_distance_numpy

distances = [[dist[0], dist[1]] for dist in NUMBA_DISTANCES]

# The key string (i.e. 'euclidean') must be the same as the name in _registry
_expected_series_results = {
    # Result structure [univariate_series, multivariate_series, multivariate_panel]
    "squared": [6.93261, 50.31911, 499.67497],
    "euclidean": [7.39956, 56.41510, 564.00440],
}


def validate_result(result: float) -> None:
    """Validate a distance result.

    Parameters
    ----------
    result: float
        Result to validate
    """
    assert isinstance(result, float)


@pytest.mark.parametrize("distance", distances)
def test_series_distances(distance: List):
    """Test all distances with series timeseries.

    Parameters
    ----------
    distance: Callable
        Distance function to run test on
    """
    name = distance[0]
    distance = distance[1]

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

    # Single result and univariate yield same result
    assert single_result == univariate_result

    # Test results are around what expected
    if name not in _expected_series_results:
        raise KeyError(
            "Please add expected test results to the "
            "_expected_series_results dict in "
            "sktime/dist_kernels/numba/tests/test_all_numba_distances.py"
        )
    expected_results = _expected_series_results[name]
    assert_almost_equal(univariate_result, expected_results[0], 3)
    assert_almost_equal(multivariate_result, expected_results[1], 3)


@pytest.mark.parametrize("distance", distances)
def test_panel_distances(distance: Callable):
    """Test all distance with panel timeseries.

    Parameters
    ----------
    distance: Callable
        Distance function to run test on
    """
    name = distance[0]
    distance = distance[1]
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

    # Test results are around what expected
    if name not in _expected_series_results:
        raise KeyError(
            "Please add expected test results to the "
            "_expected_series_results dict in "
            "sktime/dist_kernels/numba/tests/test_all_numba_distances.py"
        )

    expected_results = _expected_series_results[name]
    assert_almost_equal(univariate_result, expected_results[1], 3)
    assert_almost_equal(multivariate_result, expected_results[2], 3)
