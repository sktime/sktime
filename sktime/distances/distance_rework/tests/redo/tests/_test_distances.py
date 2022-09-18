# -*- coding: utf-8 -*-
from math import isclose

import numpy as np

from sktime.distances import distance
from sktime.distances.distance_rework.tests.redo import (
    BaseDistance,
    _DdtwDistance,
    _DtwDistance,
    _EdrDistance,
    _ErpDistance,
    _EuclideanDistance,
    _LcssDistance,
    _MsmDistance,
    _SquaredDistance,
    _TweDistance,
    _WddtwDistance,
    _WdtwDistance,
)


def _distance_tests(
    dist: BaseDistance,
    x: np.ndarray,
    y: np.ndarray,
    expected_independent: float,
    expected_dependent: float,
    use_result_callback=False,
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

    assert independent_result == dist.independent_distance(x, y)
    assert dependent_result == dist.dependent_distance(x, y)

    independent_distance_factory = dist.distance_factory(x, y, strategy="independent")
    dependent_distance_factory = dist.distance_factory(x, y, strategy="dependent")

    assert independent_result == independent_distance_factory(x, y)
    assert dependent_result == dependent_distance_factory(x, y)

    if dist._has_local_distance is True:
        local_x = 20.0
        local_y = 65.0
        local_result = dist.local_distance(local_x, local_y)
        local_distance_factory = dist.distance_factory(
            local_x, local_y, strategy="local"
        )
        assert local_result == local_distance_factory(local_x, local_y)

    # Not all distances will have a cost matrix so we disable them.
    if dist._has_cost_matrix is False:
        return

    independent_result, independent_cost_matrix = dist.distance(
        x, y, strategy="independent", return_cost_matrix=True
    )
    dependent_result, dependent_cost_matrix = dist.distance(
        x, y, strategy="dependent", return_cost_matrix=True
    )

    # Distances like edr do custom things with the final distance
    if use_result_callback is True:
        callback = dist._result_distance_callback()
        assert (
            callback(independent_cost_matrix[-1, -1], x.shape[-1], y.shape[-1])
            == independent_result
        )
        assert (
            callback(dependent_cost_matrix[-1, -1], x.shape[-1], y.shape[-1])
            == dependent_result
        )
    else:
        assert independent_cost_matrix[-1, -1] == independent_result
        assert dependent_cost_matrix[-1, -1] == dependent_result
    assert isinstance(independent_cost_matrix, np.ndarray)
    assert isinstance(dependent_cost_matrix, np.ndarray)

    # independent_path, independent_path_result = dist.distance_alignment_path(
    #     x, y, strategy="independent"
    # )
    # dependent_path, dependent_path_result = dist.distance_alignment_path(
    #     x, y, strategy="dependent"
    # )
    # assert independent_path_result == independent_result
    # assert dependent_path_result == dependent_result
    # assert isinstance(independent_path, list)
    # assert isinstance(dependent_path, list)


x_2d = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y_2d = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)

x_1d = np.array([2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13])
y_1d = np.array([5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7])

# y_2d_different_length = np.array(
#     [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]],
#     [[1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10]]
# )
#
# y_1d_different_length = np.array([5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7, 1, 2, 4, 6])


def _get_test_result(dist: BaseDistance):
    def _output_result(_x, _y, dims):
        independent_result = dist.distance(
            _x, _y, strategy="independent", return_cost_matrix=False
        )
        dependent_result = dist.distance(
            _x, _y, strategy="dependent", return_cost_matrix=False
        )

        if not isinstance(independent_result, float):
            independent_result = independent_result[0]
        if not isinstance(dependent_result, float):
            dependent_result = dependent_result[0]
        # obj_type = str(type(dist)).split(".")[-1].split("'")[0]
        # print(
        #    f"_distance_tests({obj_type}(), x_{dims}, y_{dims}, {independent_result}, "
        #     f"{dependent_result})"
        # )

    _output_result(x_1d, y_1d, "1d")
    _output_result(x_2d, y_2d, "2d")


def _check_matches_old(dist, old_ist_str):
    # 1d check
    old_distance_1d = distance(x_1d, y_1d, metric=old_ist_str, g=0.05)
    ind_1d = dist.distance(x_1d, y_1d, strategy="independent", g=0.05)
    dep_1d = dist.distance(x_1d, y_1d, strategy="dependent", g=0.05)

    old_distance_2d = distance(x_2d, y_2d, metric=old_ist_str, g=0.05)
    ind_2d = dist.distance(x_2d, y_2d, strategy="independent", g=0.05)
    dep_2d = dist.distance(x_2d, y_2d, strategy="dependent", g=0.05)

    decimal_places = 12

    def _is_close(a, b):
        return isclose(a, b, abs_tol=10**-decimal_places)

    assert _is_close(old_distance_1d, ind_1d) or _is_close(old_distance_1d, dep_1d)
    assert _is_close(old_distance_2d, ind_2d) or _is_close(old_distance_2d, dep_2d)


def test_euclidean_distance():
    """Test euclidean distance."""
    _distance_tests(
        _EuclideanDistance(), x_1d, y_1d, 123.11173786442949, 123.11173786442949
    )
    _distance_tests(
        _EuclideanDistance(), x_2d, y_2d, 66.39465339920075, 66.39465339920075
    )
    _check_matches_old(_EuclideanDistance(), "euclidean")


def test_squared_distance():
    """Test squared distance."""
    _distance_tests(_SquaredDistance(), x_1d, y_1d, 15156.5, 15156.5)
    _distance_tests(_SquaredDistance(), x_2d, y_2d, 4408.25, 4408.25)
    _check_matches_old(_SquaredDistance(), "squared")


def test_dtw():
    """Test dtw distance."""
    _distance_tests(_DtwDistance(), x_1d, y_1d, 1247.5, 1247.5)
    _distance_tests(_DtwDistance(), x_2d, y_2d, 3823.25, 4408.25)
    _check_matches_old(_DtwDistance(), "dtw")


def test_ddtw():
    """Test ddtw distance."""
    _distance_tests(_DdtwDistance(), x_1d, y_1d, 4073.984375, 4073.984375)
    _distance_tests(_DdtwDistance(), x_2d, y_2d, 3475.921875, 3833.84375)
    _check_matches_old(_DdtwDistance(), "ddtw")


def test_wdtw():
    """Test wdtw distance."""
    _distance_tests(_WdtwDistance(), x_1d, y_1d, 546.9905488055973, 546.9905488055973)
    _distance_tests(_WdtwDistance(), x_2d, y_2d, 1710.0368744540096, 1930.0354399701807)
    _check_matches_old(_WdtwDistance(), "wdtw")


def test_wddtw():
    """Test wddtw distance."""
    _distance_tests(
        _WddtwDistance(), x_1d, y_1d, 1879.3085987999807, 1879.3085987999807
    )
    _distance_tests(_WddtwDistance(), x_2d, y_2d, 1578.927748232979, 1725.86611586604)
    _check_matches_old(_WddtwDistance(), "wddtw")


def test_edr():
    """Test edr distance."""
    _distance_tests(_EdrDistance(), x_1d, y_1d, 0.3, 0.3, use_result_callback=True)
    _distance_tests(_EdrDistance(), x_2d, y_2d, 1.1, 0.6, use_result_callback=True)
    _check_matches_old(_EdrDistance(), "edr")


def test_erp():
    """Test erp distance."""
    _distance_tests(_ErpDistance(), x_1d, y_1d, 1596.5, 100.0)
    _distance_tests(_ErpDistance(), x_2d, y_2d, 3864.25, 139.26863313696143)
    _check_matches_old(_ErpDistance(), "erp")


def test_lcss():
    """Test lcss distance."""
    _distance_tests(
        _LcssDistance(),
        x_1d,
        y_1d,
        0.30000000000000004,
        0.30000000000000004,
        use_result_callback=True,
    )
    _distance_tests(
        _LcssDistance(), x_2d, y_2d, 0.19999999999999996, 0.8, use_result_callback=True
    )
    _check_matches_old(_LcssDistance(), "lcss")


def test_twe():
    """Test twe distance."""
    _distance_tests(_TweDistance(), x_1d, y_1d, 2442.018000000001, 119.01800000000003)
    _distance_tests(_TweDistance(), x_2d, y_2d, 8506.52, 277.53726627392285)
    # error in old twe so cant compare


def test_msm():
    """Test msm distance."""
    _distance_tests(_MsmDistance(), x_1d, y_1d, 65.5, 65.5)
    _distance_tests(_MsmDistance(), x_2d, y_2d, 164.5, 158.9810894318522)
    # doesnt support 2d so cant compare to old (1d is equivalent tho)
