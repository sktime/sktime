# -*- coding: utf-8 -*-
import numpy as np
import pytest
from typing import Callable

from sktime.dists_kernels.numba_aligners._registry import NUMBA_ALIGNERS
from sktime.dists_kernels.tests._utils import create_test_distance_numpy
from sktime.dists_kernels.numba_aligners.pairwise_aligners import pairwise_alignment

pairwises = [dist[2] for dist in NUMBA_ALIGNERS]


def validate_result(alignment_matrix: np.ndarray, distance_result: np.ndarray) -> None:
    """Method used to validate a result.

    Parameters
    ----------
    alignment_matrix
        [n, m, n_dims, m_dims] where n is the size of x, m is the size of y, n_dims
        is the x dimensions, m_dims is y dimensions; matrix that contains the given
        alignment calculation for each pairwise.
    distance_result
        [n, m] where n is the size of x, m is the size of y; matrix that contains the
        pairwise distances between each element.

    """
    assert isinstance(distance_result, np.ndarray)
    assert distance_result.ndim == 2
    assert isinstance(alignment_matrix, np.ndarray)
    assert alignment_matrix.ndim == 4


@pytest.mark.parametrize("pairwise", pairwises)
def test_series_pairwise_distances(pairwise: Callable):
    """Test all distances with series timeseries.

    Parameters
    ----------
    pairwise: Callable
        Distance function to run test on
    """
    x_univariate = create_test_distance_numpy(1, 1, 10)
    y_univariate = create_test_distance_numpy(1, 1, 10, random_state=2)
    univariate_alignment_matrix, univariate_pairwise_result = pairwise_alignment(
        x_univariate, y_univariate, pairwise
    )

    validate_result(univariate_alignment_matrix, univariate_pairwise_result)

    x_multivariate = create_test_distance_numpy(1, 10, 10)
    y_multivariate = create_test_distance_numpy(1, 10, 10, random_state=2)
    multivariate_alignment_matrix, multivariate_pairwise_result = pairwise_alignment(
        x_multivariate, y_multivariate, pairwise
    )
    validate_result(multivariate_alignment_matrix, multivariate_pairwise_result)


@pytest.mark.parametrize("pairwise", pairwises)
def test_panel_pairwise_distances(pairwise: Callable):
    """Test all distance with panel timeseries.

    Parameters
    ----------
    pairwise: Callable
        Distance function to run test on
    """
    x_univariate = create_test_distance_numpy(10, 1, 10)
    y_univariate = create_test_distance_numpy(10, 1, 10, random_state=2)
    univariate_alignment_matrix, univariate_pairwise_result = pairwise_alignment(
        x_univariate, y_univariate, pairwise
    )
    validate_result(univariate_alignment_matrix, univariate_pairwise_result)

    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(10, 10, 10, random_state=2)
    multivariate_alignment_matrix, multivariate_result = pairwise_alignment(
        x_multivariate, y_multivariate, pairwise
    )
    validate_result(multivariate_alignment_matrix, multivariate_result)
