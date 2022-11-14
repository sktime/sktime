# -*- coding: utf-8 -*-
"""Test for the distance module."""
import numpy as np

from sktime.distances.distance_rework_two import (
    _DdtwDistance,
    _DtwDistance,
    _EdrDistance,
    _ErpDistance,
    _LcssDistance,
    _MsmDistance,
    _SquaredDistance,
    _TweDistance,
    _WddtwDistance,
    _WdtwDistance,
)

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


def test_ddtw_distance():
    """Test ddtw."""
    dist = _DdtwDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 4073.984375
    assert dep_1d[0] == 4073.984375
    assert ind_2d[0] == 3475.921875
    assert dep_2d[0] == 3833.84375


def test_wdtw_distance():
    """Test wdtw."""
    dist = _WdtwDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 546.9905488055973
    assert dep_1d[0] == 546.9905488055973
    assert ind_2d[0] == 1710.0368744540096
    assert dep_2d[0] == 1930.0354399701807


def test_wddtw_distance():
    """Test wdtw."""
    dist = _WddtwDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 1879.3085987999807
    assert dep_1d[0] == 1879.3085987999807
    assert ind_2d[0] == 1578.927748232979
    assert dep_2d[0] == 1725.86611586604


def test_edr_distance():
    """Test edr."""
    dist = _EdrDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 0.3
    assert dep_1d[0] == 0.3
    assert ind_2d[0] == 1.2
    assert dep_2d[0] == 0.8


def test_erp_distance():
    """Test erp."""
    dist = _ErpDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 22236.5
    assert dep_1d[0] == 22236.5
    assert ind_2d[0] == 8816.5
    assert dep_2d[0] == 8816.5


def test_lcss_distance():
    """Test lcss."""
    dist = _LcssDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 0.30000000000000004
    assert dep_1d[0] == 0.30000000000000004
    assert ind_2d[0] == 0.6
    assert dep_2d[0] == 0.9


def test_twe_distance():
    """Test twe."""
    dist = _TweDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 4878.018000000002
    assert dep_1d[0] == 4878.018000000002
    assert ind_2d[0] == 17007.020000000004
    assert dep_2d[0] == 17631.0


def test_msm_distance():
    """Test msm."""
    dist = _MsmDistance()
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", return_cost_matrix=True)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", return_cost_matrix=True)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", return_cost_matrix=True)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", return_cost_matrix=True)

    assert ind_1d[0] == 146.5
    assert dep_1d[0] == 146.5
    assert ind_2d[0] == 168.0
    assert dep_2d[0] == 164.0
