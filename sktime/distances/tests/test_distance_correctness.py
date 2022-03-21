# -*- coding: utf-8 -*-
"""Test the distance calculations are correct.

Compare the distance calculations on the 1D and 2D (d,m) format input against the
results generated with tsml, LINK TO CODE.
"""
from numpy.testing import assert_almost_equal

from sktime.datasets import load_unit_test
from sktime.distances import (  # edr_distance,
    ddtw_distance,
    dtw_distance,
    erp_distance,
    lcss_distance,
    msm_distance,
    wdtw_distance,
)

distances = [
    "dtw",
    "wdtw",
    "msm",
    "ddtw",
]
distance_functions = {
    "dtw": dtw_distance,
    "wdtw": wdtw_distance,
    "msm": msm_distance,
    "ddtw": ddtw_distance
    #    "euclidean",
    #    "erp",
    #    "lcss",
    #    "ddtw",
    #    "wddtw",
}

distance_parameters = {
    "dtw": [0.0, 0.1, 1.0],  # window
    "wdtw": [0.0, 0.1, 1.0],  # parameter g
    "msm": [0.0, 0.1, 1.0],  # parameter c
    "erp": [0.0, 0.1, 1.0],  # window
    "lcss": [0.0, 50.0, 200.0],  # window
    "ddtw": [0.0, 0.1, 1.0],
}
unit_test_distances = {
    "euclidean": 0.0,
    "dtw": [384147.0, 315012.0, 275854.0],
    "wdtw": [137927.0, 68406.15849, 2.2296],
    "msm": [1515.0, 1516.4, 1529.0],
    "erp": [168.0, 1172.0, 2275.0],
    "lcss": [1.0, 0.45833, 0.08333],
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


def test_correctness():
    """Test dtw correctness."""
    trainX, trainy = load_unit_test(return_type="numpy3D")
    trainX2, trainy2 = load_unit_test(return_type="numpy2D")
    #    trainX3, trainy3 = load_basic_motions(return_type="numpy3D")
    # Test instances from UnitTest
    cases1 = [trainX[0], trainX2[0]]
    cases2 = [trainX[2], trainX2[2]]
    # Add test cases1 and 2 are the same
    for i in range(0, len(cases1)):
        for j in range(0, 3):
            d = dtw_distance(cases1[i], cases2[i], window=distance_parameters["dtw"][j])
            assert_almost_equal(d, unit_test_distances["dtw"][j], 4)
            d = wdtw_distance(cases1[i], cases2[i], g=distance_parameters["wdtw"][j])
            assert_almost_equal(d, unit_test_distances["wdtw"][j], 4)
            d = msm_distance(cases1[i], cases2[i], c=distance_parameters["msm"][j])
            assert_almost_equal(d, unit_test_distances["msm"][j], 4)
            d = erp_distance(cases1[i], cases2[i], window=distance_parameters["erp"][j])
            assert_almost_equal(d, unit_test_distances["erp"][j], 4)
            d = lcss_distance(
                cases1[i], cases2[i], epsilon=distance_parameters["lcss"][j]
            )
            assert_almost_equal(d, unit_test_distances["lcss"][j], 4)
