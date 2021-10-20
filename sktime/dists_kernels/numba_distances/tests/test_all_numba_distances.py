# -*- coding: utf-8 -*-
import pytest
from typing import Callable

from sktime.dists_kernels.numba_distances._registry import NUMBA_DISTANCES
from sktime.dists_kernels.numba_distances.tests.utils import create_test_distance_numpy

distances = [dist[1] for dist in NUMBA_DISTANCES]


def validate_result(result: float) -> None:
    """Method used to validate a result.

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
    x_univariate = create_test_distance_numpy(1, 1, 10)
    y_univariate = create_test_distance_numpy(1, 1, 10, random_state=2)
    univariate_result = distance(x_univariate, y_univariate)
    validate_result(univariate_result)

    x_multivariate = create_test_distance_numpy(1, 10, 10)
    y_multivariate = create_test_distance_numpy(1, 10, 10, random_state=2)
    multivariate_result = distance(x_multivariate, y_multivariate)
    validate_result(multivariate_result)


@pytest.mark.parametrize("distance", distances)
def test_panel_distances(distance: Callable):
    """Test all distance with panel timeseries.

    Parameters
    ----------
    distance: Callable
        Distance function to run test on
    """
    x_univariate = create_test_distance_numpy(10, 1, 10)
    y_univariate = create_test_distance_numpy(10, 1, 10, random_state=2)
    univariate_result = distance(x_univariate, y_univariate)
    validate_result(univariate_result)

    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(10, 10, 10, random_state=2)
    multivariate_result = distance(x_multivariate, y_multivariate)
    validate_result(multivariate_result)
