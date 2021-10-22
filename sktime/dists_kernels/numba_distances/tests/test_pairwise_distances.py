# -*- coding: utf-8 -*-
import numpy as np
import pytest
from typing import Callable

from sktime.dists_kernels.numba_distances._elastic._registry import NUMBA_DISTANCES
from sktime.dists_kernels.tests._utils import create_test_distance_numpy

pairwises = [dist[2] for dist in NUMBA_DISTANCES]


def validate_result(result: np.ndarray) -> None:
    """Method used to validate a result.

    Parameters
    ----------
    result: np.ndarray
        [n, m] matrix with pairwise results
    """
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


@pytest.mark.parametrize("pairwise", pairwises)
def test_series_pairwise_distances(pairwise: Callable):
    """Test all distances with series timeseries.

    Parameters
    ----------
    pairwise: Callable
        Distance function to run test on
    """
    if pairwise is None:
        return
    x_univariate = create_test_distance_numpy(1, 1, 10)
    y_univariate = create_test_distance_numpy(1, 1, 10, random_state=2)
    univariate_pairwise_result = pairwise(x_univariate, y_univariate)

    validate_result(univariate_pairwise_result)

    x_multivariate = create_test_distance_numpy(1, 10, 10)
    y_multivariate = create_test_distance_numpy(1, 10, 10, random_state=2)
    multivariate_pairwise_result = pairwise(x_multivariate, y_multivariate)
    validate_result(multivariate_pairwise_result)


@pytest.mark.parametrize("pairwise", pairwises)
def test_panel_pairwise_distances(pairwise: Callable):
    """Test all distance with panel timeseries.

    Parameters
    ----------
    pairwise: Callable
        Distance function to run test on
    """
    if pairwise is None:
        return
    x_univariate = create_test_distance_numpy(10, 1, 10)
    y_univariate = create_test_distance_numpy(10, 1, 10, random_state=2)
    univariate_pairwise_result = pairwise(x_univariate, y_univariate)
    validate_result(univariate_pairwise_result)

    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(10, 10, 10, random_state=2)
    multivariate_pairwise_result = pairwise(x_multivariate, y_multivariate)
    validate_result(multivariate_pairwise_result)
