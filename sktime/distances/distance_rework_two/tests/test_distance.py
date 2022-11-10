# -*- coding: utf-8 -*-
"""Test for the distance module."""
import numpy as np

from sktime.distances.distance_rework_two import _DtwDistance, _SquaredDistance

x_2d = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y_2d = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)

x_1d = np.array([2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13])
y_1d = np.array([5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7])


def test_squared_distance():
    """Test squared."""
    dist = _SquaredDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent")
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent")
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent")
    local = dist.distance(x_1d[0], y_1d[0], strategy="local")

    assert ind_1d == 15156.5
    assert dep_1d == 15156.5
    assert ind_2d == 4408.25
    assert dep_2d == 4408.25
    assert local == 9.0


def test_dtw_distance():
    """Test dtw."""
    dist = _DtwDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 1247.5
    assert ind_2d[0] == 3823.25
    assert dep_1d[0] == 1247.5
    assert dep_2d[0] == 4408.25
