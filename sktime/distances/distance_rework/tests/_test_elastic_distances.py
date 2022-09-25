# -*- coding: utf-8 -*-
import numpy as np

from sktime.distances.distance_rework import (
    ElasticDistance,
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

def _alignment_test(
        dist: ElasticDistance,
        x: np.ndarray,
        y: np.ndarray,
):
    def _run_alignment_test(strategy: str):
        alignment = dist.alignment_path(x, y, strategy=strategy)
        alignment_1, distance = dist.alignment_path(x, y, strategy=strategy, return_distance=True)
        alignment_2, distance_1, cost_matrix = dist.alignment_path(x, y, strategy=strategy, return_distance=True, return_cost_matrix=True)
        callable_func = dist.alignment_path_factory(x, y, strategy=strategy, return_distance=True, return_cost_matrix=True)
        alignment_3, distance_2, cost_matrix_1 = callable_func(x, y)
        assert isinstance(alignment, list)
        assert isinstance(alignment[0], tuple)
        assert isinstance(distance, float)
        assert isinstance(cost_matrix, np.ndarray)
        assert alignment == alignment_1
        assert alignment_1 == alignment_2
        assert distance == distance_1
        assert np.array_equal(cost_matrix, cost_matrix_1)
        assert distance == distance_2
        assert alignment == alignment_3
    _run_alignment_test('independent')
    _run_alignment_test('dependent')


x_2d = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y_2d = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)

x_1d = np.array([2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13])
y_1d = np.array([5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7])


def test_dtw():
    _alignment_test(_DtwDistance(), x_1d, y_1d)
    _alignment_test(_DtwDistance(), x_2d, y_2d)


def test_ddtw():
    _alignment_test(_DdtwDistance(), x_1d, y_1d)
    _alignment_test(_DdtwDistance(), x_2d, y_2d)


def test_wdtw():
    _alignment_test(_WdtwDistance(), x_1d, y_1d)
    _alignment_test(_WdtwDistance(), x_2d, y_2d)


def test_wddtw():
    _alignment_test(_WddtwDistance(), x_1d, y_1d)
    _alignment_test(_WddtwDistance(), x_2d, y_2d)


def test_edr():
    _alignment_test(_EdrDistance(), x_1d, y_1d)
    _alignment_test(_EdrDistance(), x_2d, y_2d)


def test_erp():
    _alignment_test(_ErpDistance(), x_1d, y_1d)
    _alignment_test(_ErpDistance(), x_2d, y_2d)


def test_msm():
    _alignment_test(_MsmDistance(), x_1d, y_1d)
    _alignment_test(_MsmDistance(), x_2d, y_2d)


def test_twe():
    _alignment_test(_TweDistance(), x_1d, y_1d)
    _alignment_test(_TweDistance(), x_2d, y_2d)


def test_lcss():
    _alignment_test(_LcssDistance(), x_1d, y_1d)
    _alignment_test(_LcssDistance(), x_2d, y_2d)
