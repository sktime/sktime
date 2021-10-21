# -*- coding: utf-8 -*-
import numpy as np
import pytest
from typing import Callable

from sktime.dists_kernels.numba_distances._elastic._registry import NUMBA_DISTANCES
from sktime.dists_kernels.tests._utils import create_test_distance_numpy
from sktime.dists_kernels.numba_distances.pairwise_distances import pairwise_distance

distances = [dist[2] for dist in NUMBA_DISTANCES]


def validate_result(result: np.ndarray) -> None:
    """Method used to validate a result.

    Parameters
    ----------
    result: np.ndarray
        [n, m] matrix with pairwise results
    """
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


@pytest.mark.parametrize("distance", distances)
def test_series_pairwise_distances(distance: Callable):
    """Test all distances with series timeseries.

    Parameters
    ----------
    distance: Callable
        Distance function to run test on
    """
    x_univariate = create_test_distance_numpy(1, 1, 10)
    y_univariate = create_test_distance_numpy(1, 1, 10, random_state=2)
    univariate_pairwise_result = pairwise_distance(x_univariate, y_univariate, distance)

    validate_result(univariate_pairwise_result)

    x_multivariate = create_test_distance_numpy(1, 10, 10)
    y_multivariate = create_test_distance_numpy(1, 10, 10, random_state=2)
    multivariate_pairwise_result = pairwise_distance(
        x_multivariate, y_multivariate, distance
    )
    validate_result(multivariate_pairwise_result)


@pytest.mark.parametrize("distance", distances)
def test_panel_pairwise_distances(distance: Callable):
    """Test all distance with panel timeseries.

    Parameters
    ----------
    distance: Callable
        Distance function to run test on
    """
    x_univariate = create_test_distance_numpy(10, 1, 10)
    y_univariate = create_test_distance_numpy(10, 1, 10, random_state=2)
    univariate_pairwise_result = pairwise_distance(x_univariate, y_univariate, distance)
    validate_result(univariate_pairwise_result)

    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(10, 10, 10, random_state=2)
    multivariate_result = pairwise_distance(x_multivariate, y_multivariate, distance)
    validate_result(multivariate_result)
