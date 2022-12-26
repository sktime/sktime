# -*- coding: utf-8 -*-
"""Test the distance calculations are correct.

Compare the distance calculations on the 1D and 2D (d,m) format input against the
results generated with tsml, in distances.tests.TestDistances.
"""
__author__ = ["chrisholder", "TonyBagnall"]

import pytest
from numpy.testing import assert_almost_equal

from sktime.datasets import load_basic_motions, load_unit_test
from sktime.distances import (
    ddtw_distance,
    dtw_distance,
    edr_distance,
    erp_distance,
    euclidean_distance,
    lcss_distance,
    msm_distance,
    twe_distance,
    wddtw_distance,
    wdtw_distance,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

distances = [
    "dtw",
    "wdtw",
    "lcss",
    "msm",
    "ddtw",
    "euclidean",
    "erp",
    "ddtw",
    "wddtw",
    "twe",
]

distance_parameters = {
    "dtw": [0.0, 0.1, 1.0],  # window
    "wdtw": [0.0, 0.1, 1.0],  # parameter g
    "wddtw": [0.0, 0.1, 1.0],  # parameter g
    "msm": [0.0, 0.1, 1.0],  # parameter c
    "erp": [0.0, 0.1, 1.0],  # window
    "lcss": [0.0, 50.0, 200.0],  # espilon
    "edr": [0.0, 50.0, 200.0],  # espilon
    "ddtw": [0.0, 0.1, 1.0],  # window
    "twe": [0.0, 0.1, 1.0],  # window
}
unit_test_distances = {
    "euclidean": 619.7959,
    "dtw": [384147.0, 315012.0, 275854.0],
    "wdtw": [137927.0, 68406.15849, 2.2296],
    "msm": [1515.0, 1516.4, 1529.0],
    "erp": [168.0, 1107.0, 2275.0],
    "lcss": [1.0, 0.45833, 0.08333],
    "edr": [1.0, 0.58333, 0.125],
    "ddtw": [80806.0, 76289.0625, 76289.0625],
    "wddtw": [38144.53125, 19121.4927, 1.34957],
    "twe": [242.001, 628.0029999999999, 3387.044],
}
basic_motions_distances = {
    "euclidean": 27.51835240,
    "dtw": [757.259719, 330.834497, 330.834497],
    "wdtw": [165.41724, 3.308425, 0],
    "msm": [70.014828, 89.814828, 268.014828],
    "erp": [0.2086269, 2.9942540, 102.097904],
    "edr": [1.0, 0.26, 0.07],
    "lcss": [1.0, 0.26, 0.05],
    "ddtw": [297.18771, 160.48649, 160.29823],
    "wddtw": [80.149117, 1.458858, 0.0],
    "twe": [1.325876246546281, 14.759114523578294, 218.21301289250758],
}


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_multivariate_correctness():
    """Test distance correctness on BasicMotions: multivariate, equal length."""
    trainX, trainy = load_basic_motions(return_type="numpy3D")
    case1 = trainX[0]
    case2 = trainX[1]
    d = euclidean_distance(case1, case2)
    assert_almost_equal(d, basic_motions_distances["euclidean"], 4)
    twe_mult = []
    for j in range(0, 3):
        d = dtw_distance(case1, case2, window=distance_parameters["dtw"][j])
        assert_almost_equal(d, basic_motions_distances["dtw"][j], 4)
        d = wdtw_distance(case1, case2, g=distance_parameters["wdtw"][j])
        assert_almost_equal(d, basic_motions_distances["wdtw"][j], 4)
        d = lcss_distance(case1, case2, epsilon=distance_parameters["lcss"][j] / 50.0)
        assert_almost_equal(d, basic_motions_distances["lcss"][j], 4)
        d = erp_distance(case1, case2, window=distance_parameters["erp"][j])
        assert_almost_equal(d, basic_motions_distances["erp"][j], 4)
        d = edr_distance(case1, case2, epsilon=distance_parameters["edr"][j] / 50.0)
        assert_almost_equal(d, basic_motions_distances["edr"][j], 4)
        d = ddtw_distance(case1, case2, window=distance_parameters["ddtw"][j])
        assert_almost_equal(d, basic_motions_distances["ddtw"][j], 4)
        d = wddtw_distance(case1, case2, g=distance_parameters["wddtw"][j])
        assert_almost_equal(d, basic_motions_distances["wddtw"][j], 4)
        d = twe_distance(case1, case2, window=distance_parameters["twe"][j])
        twe_mult.append(d)
        assert_almost_equal(d, basic_motions_distances["twe"][j], 4)


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_univariate_correctness():
    """Test dtw correctness on UnitTest: univariate, equal length."""
    trainX, trainy = load_unit_test(return_type="numpy3D")
    trainX2, trainy2 = load_unit_test(return_type="numpy2D")
    # Test 2D and 3D instances from UnitTest
    cases1 = [trainX[0], trainX2[0]]
    cases2 = [trainX[2], trainX2[2]]
    # Add test cases1 and 2 are the same
    d = euclidean_distance(cases1[0], cases2[0])
    d2 = euclidean_distance(cases1[1], cases2[1])
    assert_almost_equal(d, unit_test_distances["euclidean"], 4)
    assert d == d2
    twe_uni = []
    for j in range(0, 3):
        d = dtw_distance(cases1[0], cases2[0], window=distance_parameters["dtw"][j])
        d2 = dtw_distance(cases1[1], cases2[1], window=distance_parameters["dtw"][j])
        assert_almost_equal(d, unit_test_distances["dtw"][j], 4)
        assert d == d2
        d = wdtw_distance(cases1[0], cases2[0], g=distance_parameters["wdtw"][j])
        d2 = wdtw_distance(cases1[1], cases2[1], g=distance_parameters["wdtw"][j])
        assert_almost_equal(d, unit_test_distances["wdtw"][j], 4)
        assert d == d2
        d = lcss_distance(cases1[0], cases2[0], epsilon=distance_parameters["lcss"][j])
        d2 = lcss_distance(cases1[1], cases2[1], epsilon=distance_parameters["lcss"][j])
        assert_almost_equal(d, unit_test_distances["lcss"][j], 4)
        assert d == d2
        d = msm_distance(cases1[0], cases2[0], c=distance_parameters["msm"][j])
        d2 = msm_distance(cases1[1], cases2[1], c=distance_parameters["msm"][j])
        assert_almost_equal(d, unit_test_distances["msm"][j], 4)
        assert d == d2
        d = erp_distance(cases1[0], cases2[0], window=distance_parameters["erp"][j])
        d2 = erp_distance(cases1[1], cases2[1], window=distance_parameters["erp"][j])
        assert_almost_equal(d, unit_test_distances["erp"][j], 4)
        assert d == d2
        d = edr_distance(cases1[0], cases2[0], epsilon=distance_parameters["edr"][j])
        d2 = edr_distance(cases1[1], cases2[1], epsilon=distance_parameters["edr"][j])
        assert_almost_equal(d, unit_test_distances["edr"][j], 4)
        assert d == d2
        d = ddtw_distance(cases1[0], cases2[0], window=distance_parameters["ddtw"][j])
        d2 = ddtw_distance(cases1[1], cases2[1], window=distance_parameters["ddtw"][j])
        assert_almost_equal(d, unit_test_distances["ddtw"][j], 4)
        assert d == d2
        d = wddtw_distance(cases1[0], cases2[0], g=distance_parameters["wddtw"][j])
        d2 = wddtw_distance(cases1[1], cases2[1], g=distance_parameters["wddtw"][j])
        assert_almost_equal(d, unit_test_distances["wddtw"][j], 4)
        assert d == d2
        d = twe_distance(cases1[0], cases2[1], window=distance_parameters["twe"][j])
        d2 = twe_distance(cases1[1], cases2[1], window=distance_parameters["twe"][j])
        twe_uni.append(d)
        assert_almost_equal(d, unit_test_distances["twe"][j], 4)
        assert d == d2
