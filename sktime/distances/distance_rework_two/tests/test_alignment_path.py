# -*- coding: utf-8 -*-
"""Test for the distance alignment"""
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


def make_assertions(ind_1d, dep_1d, ind_2d, dep_2d):
    print("\n\n")
    print(f"assert ind_1d == {ind_1d}")
    print(f"assert dep_1d == {dep_1d}")
    print(f"assert ind_2d == {ind_2d}")
    print(f"assert dep_2d == {dep_2d}")


def test_dtw_distance():
    """Test dtw alignment."""
    dist = _DtwDistance()
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    assert ind_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert dep_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert ind_2d == [
        (0, 0),
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]
    assert dep_2d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]


def test_ddtw_distance():
    """Test ddtw alignment."""
    dist = _DdtwDistance()
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    assert ind_1d == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 1),
        (5, 2),
        (6, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (7, 7),
    ]
    assert dep_1d == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 1),
        (5, 2),
        (6, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (7, 7),
    ]
    assert ind_2d == [
        (0, 0),
        (1, 0),
        (2, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
    ]
    assert dep_2d == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]


def test_wdtw_distance():
    """Test wdtw alignment."""
    dist = _WdtwDistance()
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    assert ind_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert dep_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert ind_2d == [
        (0, 0),
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]
    assert dep_2d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]


def test_wddtw_distance():
    """Test wddtw alignment."""
    dist = _WddtwDistance()
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    assert ind_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert dep_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert ind_2d == [
        (0, 0),
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]
    assert dep_2d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]


def test_erp_distance():
    """Test erp alignment."""
    dist = _ErpDistance()
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    assert ind_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert dep_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert ind_2d == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
        (5, 8),
        (6, 8),
        (7, 8),
        (8, 8),
        (9, 9),
    ]
    assert dep_2d == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]


def test_msm_distance():
    """Test msm alignment."""
    dist = _MsmDistance()
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    assert ind_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 5),
        (6, 5),
        (7, 5),
        (8, 6),
        (9, 7),
        (9, 8),
        (9, 9),
    ]
    assert dep_1d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert ind_2d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]
    assert dep_2d == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]


def test_twe_distance():
    """Test twe alignment."""
    dist = _TweDistance()
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    assert ind_1d == [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert dep_1d == [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert ind_2d == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 9),
        (6, 9),
        (7, 9),
        (8, 9),
        (9, 9),
    ]
    assert dep_2d == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
    ]


def test_lcss_distance():
    """Test lcss alignment."""
    dist = _LcssDistance()
    dist._numba_distance = False
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    make_assertions(ind_1d, dep_1d, ind_2d, dep_2d)


def test_edr_distance():
    """Test edr alignment."""
    dist = _EdrDistance()
    dist._numba_distance = False
    ind_1d = dist.alignment_path(x_1d, y_1d, strategy="independent")
    dep_1d = dist.alignment_path(x_1d, y_1d, strategy="dependent")
    ind_2d = dist.alignment_path(x_2d, y_2d, strategy="independent")
    dep_2d = dist.alignment_path(x_2d, y_2d, strategy="dependent")

    make_assertions(ind_1d, dep_1d, ind_2d, dep_2d)
