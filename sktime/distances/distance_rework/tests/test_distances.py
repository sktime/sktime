# -*- coding: utf-8 -*-
"""Test for distance module."""
# -*- coding: utf-8 -*-
import numpy as np
from numba import config

from sktime.distances.distance_rework import (
    _DtwDistance,
    _EuclideanDistance,
    _SquaredEuclidean,
    _DdtwDistance,
    _WdtwDistance,
    _WddtwDistance
)
from sktime.distances.distance_rework.base import BaseDistance

config.DISABLE_JIT = False


def _distance_tests(
        dist: BaseDistance,
        x: np.ndarray,
        y: np.ndarray,
        expected_independent: float,
        expected_dependent: float
):
    """Test a BaseDistance object.

    Parameters
    ----------
    dist : BaseDistance
        The distance object to test.
    x : np.ndarray
        The first time series.
    y : np.ndarray
        The second time series.
    expected_independent : float
        The expected result for the independent strategy.
    expected_dependent : float
        The expected result for the dependent strategy.
    """
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    assert independent_result == expected_independent
    assert dependent_result == expected_dependent

    independent_cost_matrix, independent_result = dist.distance(
        x, y, strategy="independent", return_cost_matrix=True
    )
    dependent_cost_matrix, dependent_result = dist.distance(
        x, y, strategy="dependent", return_cost_matrix=True
    )
    assert isinstance(independent_cost_matrix, np.ndarray)
    assert independent_cost_matrix[-1, -1] == independent_result
    assert isinstance(dependent_cost_matrix, np.ndarray)
    assert dependent_cost_matrix[-1, -1] == dependent_result

    independent_path, independent_path_result = dist.distance_alignment_path(x, y,
                                                                             strategy="independent")
    dependent_path, dependent_path_result = dist.distance_alignment_path(x, y,
                                                                         strategy="dependent")
    assert independent_path_result == independent_result
    assert dependent_path_result == dependent_result
    assert isinstance(independent_path, list)
    assert isinstance(dependent_path, list)

x = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)

def _get_test_result(dist: BaseDistance):
    """Utility method to get the results of a distance test."""
    print("\n")
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    obj_type = str(type(dist)).split('.')[-1].split("'")[0]
    print(f'_distance_tests({obj_type}(), x, y, {independent_result}, {dependent_result})')

def test_euclidean_distance():
    dist = _EuclideanDistance()
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    assert independent_result == 66.39465339920075
    assert dependent_result == 66.39465339920075

def test_squared_distance():
    dist = _SquaredEuclidean()
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    assert independent_result == 4408.25
    assert dependent_result == 4408.25


def test_dtw_distance():
    """Test dtw distance."""
    _distance_tests(_DtwDistance(), x, y, 3823.25, 4408.25)


def test_ddtw_distance():
    """Test ddtw distance."""
    _get_test_result(_DdtwDistance())


def test_wdtw_distance():
    """Test wdtw distance."""
    dist = _WdtwDistance()

    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    assert independent_result == 1792.7229179752326
    assert dependent_result == 1930.0354399701807

    independent_cost_matrix, independent_result = dist.distance(
        x, y, strategy="independent", return_cost_matrix=True
    )
    dependent_cost_matrix, dependent_result = dist.distance(
        x, y, strategy="dependent", return_cost_matrix=True
    )
    assert isinstance(independent_cost_matrix, np.ndarray)
    assert independent_cost_matrix[-1, -1] == independent_result
    assert isinstance(dependent_cost_matrix, np.ndarray)
    assert dependent_cost_matrix[-1, -1] == dependent_result


def test_wddtw_distance():
    """Test wddtw distance."""
    dist = _WddtwDistance()

    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")

    assert independent_result == 1578.9277482329787
    assert dependent_result == 1916.921875

    independent_cost_matrix, independent_result = dist.distance(
        x, y, strategy="independent", return_cost_matrix=True
    )
    dependent_cost_matrix, dependent_result = dist.distance(
        x, y, strategy="dependent", return_cost_matrix=True
    )
    assert isinstance(independent_cost_matrix, np.ndarray)
    assert independent_cost_matrix[-1, -1] == independent_result
    assert isinstance(dependent_cost_matrix, np.ndarray)
    assert dependent_cost_matrix[-1, -1] == dependent_result
