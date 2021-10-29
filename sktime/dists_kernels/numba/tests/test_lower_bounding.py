# -*- coding: utf-8 -*-
"""Test suite for lower bounding techniques."""
import numpy as np

from sktime.dists_kernels.numba.distances.dtw_based.lower_bounding import LowerBounding
from sktime.dists_kernels.tests._utils import create_test_distance_numpy


def _validate_bounding(matrix: np.ndarray):
    assert isinstance(matrix, np.ndarray)
    assert matrix.ndim == 2


def test_lower_bounding():
    """Test for various lower bounding methods."""
    x = create_test_distance_numpy(10, 10)
    y = create_test_distance_numpy(15, 10, random_state=2)

    no_constraints = LowerBounding.NO_BOUNDING

    nb_matrix = no_constraints.create_bounding_matrix(x, y)
    _validate_bounding(nb_matrix)

    sakoe_chiba = LowerBounding.SAKOE_CHIBA

    sc_matrix = sakoe_chiba.create_bounding_matrix(x, y, sakoe_chiba_window_radius=2)
    _validate_bounding(sc_matrix)

    itakura_parallelogram = LowerBounding.ITAKURA_PARALLELOGRAM
    ip_matrix = itakura_parallelogram.create_bounding_matrix(x, y)
    _validate_bounding(ip_matrix)

    x_multivariate = create_test_distance_numpy(10, 10, 10)
    y_multivariate = create_test_distance_numpy(15, 10, 10, random_state=2)

    no_constraints = LowerBounding.NO_BOUNDING

    no_constraints.create_bounding_matrix(x_multivariate, y_multivariate)
