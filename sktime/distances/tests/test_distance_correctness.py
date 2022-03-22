# -*- coding: utf-8 -*-
"""Test the distance calculations are correct.

Compare the distance calculations on the 1D and 2D (d,m) format input against the
results generated with tsml, LINK TO CODE.
"""
from sktime.datasets import load_basic_motions, load_unit_test
from sktime.distances import dtw_distance

distance_functions = [
    "euclidean",
    "dtw",
    "wdtw",
    "msm",
    "erp",
    "lcss",
    "ddtw",
    "wddtw",
]

unit_test_distances = {
    "euclidean": 0.0,
    "dtw": [275854.0, 315012.0, 384147.0],
    "wdtw": 0.0,
    "msm": 0.0,
    "erp": 0.0,
    "lcss": 0.0,
    "ddtw": 0.0,
    "wddtw": 0.0,
}
basic_motions_distances = {
    "euclidean": 0.0,
    "dtw": 0.0,
    "wdtw": 0.0,
    "msm": 0.0,
    "erp": 0.0,
    "lcss": 0.0,
    "ddtw": 0.0,
    "wddtw": 0.0,
}


def test_dtw():
    """Test dtw correctness."""
    trainX, trainy = load_unit_test(return_type="numpy3D")
    d = dtw_distance(trainX[0], trainX[1], window=0.01)
    from sktime.distances import squared_distance
    test = squared_distance(trainX[0], trainX[1])
    joe = ''
    assert d == unit_test_distances["dtw"][0]
    d = dtw_distance(trainX[0], trainX[2], window=0.1)
    assert d == unit_test_distances["dtw"][1]
    d = dtw_distance(trainX[0], trainX[2], window=0.0)
    assert d == unit_test_distances["dtw"][2]
    trainX, trainy = load_unit_test(return_type="numpy2D")
    d = dtw_distance(trainX[0], trainX[2], window=1.0)
    assert d == unit_test_distances["dtw"][0]
    d = dtw_distance(trainX[0], trainX[2], window=0.1)
    assert d == unit_test_distances["dtw"][1]
    d = dtw_distance(trainX[0], trainX[2], window=0.0)
    assert d == unit_test_distances["dtw"][2]
    trainX, trainy = load_basic_motions(return_type="numpy3D")
    d = dtw_distance(trainX[0], trainX[2], window=1.0)
    assert d == unit_test_distances["dtw"][0]
    d = dtw_distance(trainX[0], trainX[2], window=0.1)
    assert d == unit_test_distances["dtw"][1]
    d = dtw_distance(trainX[0], trainX[2], window=0.0)
