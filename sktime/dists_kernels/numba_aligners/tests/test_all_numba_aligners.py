# -*- coding: utf-8 -*-
import pytest
from typing import Callable
import numpy as np

from sktime.dists_kernels.numba_aligners._registry import NUMBA_ALIGNERS
from sktime.dists_kernels.tests._utils import create_test_distance_numpy

aligners = [dist[1] for dist in NUMBA_ALIGNERS]


def validate_result(alignment_matrix: np.ndarray, distance_result: float) -> None:
    """Method used to validate a result.

    Parameters
    ----------
    alignment_matrix: np.ndarray
        [n, m] where n is the size of x, m is the size of y; matrix that contains the
        pairwise distances between each element.
    distance_result: float
        Distance between the two time series
    """
    assert isinstance(distance_result, float)
    assert isinstance(alignment_matrix, np.ndarray)
    assert alignment_matrix.ndim == 2


@pytest.mark.parametrize("aligner", aligners)
def test_series_distances(aligner: Callable):
    """Test all distances with series timeseries.

    Parameters
    ----------
    aligner: Callable
        Distance function to run test on
    """
    x_univariate = create_test_distance_numpy(1, 1, 10)
    y_univariate = create_test_distance_numpy(1, 1, 10, random_state=2)
    univariate_alignment_matrix, univariate_pairwise_result = aligner(
        x_univariate, y_univariate
    )
    validate_result(univariate_alignment_matrix, univariate_pairwise_result)

    x_multivariate = create_test_distance_numpy(1, 10, 10)
    y_multivariate = create_test_distance_numpy(1, 10, 10, random_state=2)
    multivariate_alignment_matrix, multivariate_pairwise_result = aligner(
        x_multivariate, y_multivariate
    )
    validate_result(multivariate_alignment_matrix, multivariate_pairwise_result)


@pytest.mark.parametrize("aligner", aligners)
def test_panel_distances(aligner: Callable):
    """Test all distance with panel timeseries.

    Parameters
    ----------
    aligner: Callable
        Distance function to run test on
    """
    x_univariate = create_test_distance_numpy(10, 1, 10)
    y_univariate = create_test_distance_numpy(10, 1, 10, random_state=2)
    univariate_alignment_matrix, univariate_pairwise_result = aligner(
        x_univariate, y_univariate
    )
    validate_result(univariate_alignment_matrix, univariate_pairwise_result)

    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(10, 10, 10, random_state=2)
    multivariate_alignment_matrix, multivariate_pairwise_result = aligner(
        x_multivariate, y_multivariate
    )
    validate_result(multivariate_alignment_matrix, multivariate_pairwise_result)
