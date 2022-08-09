# -*- coding: utf-8 -*-
"""Test for distance module."""
# -*- coding: utf-8 -*-
import numpy as np
from numba import config

from sktime.distances.distance_rework import (
    _DtwDistance,
    _EuclideanDistance,
    _SquaredEuclidean,
    _DdtwDistance
)
from sktime.distances.tests._utils import create_test_distance_numpy

config.DISABLE_JIT = True


def test_euclidiean_distances():
    """Test euclidean distance."""
    series = create_test_distance_numpy(2, 10, 10)
    x = series[0]
    y = series[1]

    dist = _EuclideanDistance()
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")

    assert independent_result == 6.1463084302283395
    assert dependent_result == 6.1463084302283395


def test_squared_euclidean_distances():
    """Test squared euclidean distance."""
    series = create_test_distance_numpy(2, 10, 10)
    x = series[0]
    y = series[1]

    dist = _SquaredEuclidean()
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")

    assert independent_result == 37.777107319495954
    assert dependent_result == 37.777107319495954

x = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)

def test_dtw_distance():
    """Test dtw distance."""


    dist = _DtwDistance()

    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    assert independent_result == 3823.25
    assert dependent_result == 4408.25

    independent_cost_matrix, independent_result = dist.distance(
        x, y, strategy="independent", return_cost_matrix=True
    )
    dependent_cost_matrix, dependent_result = dist.distance(
        x, y, strategy="dependent", return_cost_matrix=True
    )
    assert isinstance(independent_cost_matrix, np.ndarray)
    assert isinstance(dependent_cost_matrix, np.ndarray)

def test_ddtw_distance():
    """Test ddtw distance."""

    dist = _DdtwDistance()

    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    assert independent_result == 3475.921875
    assert dependent_result == 3833.84375

    independent_cost_matrix, independent_result = dist.distance(
        x, y, strategy="independent", return_cost_matrix=True
    )
    dependent_cost_matrix, dependent_result = dist.distance(
        x, y, strategy="dependent", return_cost_matrix=True
    )
    assert isinstance(independent_cost_matrix, np.ndarray)
    assert isinstance(dependent_cost_matrix, np.ndarray)
