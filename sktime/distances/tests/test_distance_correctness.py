"""Test the distance calculations are correct.

Compare the distance calculations on the 1D and 2D (d,m) format input against the
results generated with tsml, in distances.tests.TestDistances.
"""
__author__ = ["chrisholder", "TonyBagnall", "fkiraly"]

import pytest
from numpy.testing import assert_almost_equal

from sktime.datasets import load_basic_motions, load_unit_test
from sktime.distances._distance import _METRIC_INFOS
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.validation._dependencies import _check_soft_dependencies

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
distance_param_name = {
    "dtw": "window",
    "wdtw": "g",
    "wddtw": "g",
    "msm": "c",
    "erp": "window",
    "lcss": "epsilon",
    "edr": "epsilon",
    "ddtw": "window",
    "twe": "window",
}
unit_test_distances = {
    "euclidean": [619.7959],
    "dtw": [384147.0, 315012.0, 275854.0],
    "wdtw": [137927.0, 68406.15849, 2.2296],
    "msm": [1515.0, 1516.4, 1529.0],
    "erp": [168.0, 1107.0, 2275.0],
    "lcss": [1.0, 0.45833, 0.08333],
    "edr": [1.0, 0.58333, 0.125],
    "ddtw": [80806.0, 76289.0625, 76289.0625],
    "wddtw": [38144.53125, 19121.4927, 1.34957],
    "twe": [242.001, 628.0029999999999, 3387.044],
    "squared": [384147.0],
}
basic_motions_distances = {
    "euclidean": [27.51835240],
    "dtw": [757.259719, 330.834497, 330.834497],
    "wdtw": [165.41724, 3.308425, 0],
    "msm": [70.014828, 89.814828, 268.014828],
    "erp": [0.2086269, 2.9942540, 102.097904],
    "edr": [1.0, 0.26, 0.07],
    "lcss": [1.0, 0.26, 0.05],
    "ddtw": [297.18771, 160.48649, 160.29823],
    "wddtw": [80.149117, 1.458858, 0.0],
    "twe": [1.325876246546281, 14.759114523578294, 218.21301289250758],
    "squared": [757.259719],
}


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("uni_multi", ["uni", "multi"])
@pytest.mark.parametrize("dist", _METRIC_INFOS)
def test_correctness(dist, uni_multi):
    """Test dtw correctness on:

    uni_multi = "uni" -> UnitTest: univariate, equal length.
    uni_multi = "multi" -> BasicMotions: multivariate, equal length.
    """
    # skip test if distance function/class have not changed
    if not run_test_for_class([dist.dist_func, dist.dist_instance.__class__]):
        return None

    # msm distance is not implemented for multivariate
    if uni_multi == "multi" and dist.canonical_name in ["msm"]:
        return None

    dist_str = dist.canonical_name
    dist = dist.dist_func

    if uni_multi == "uni":
        loader = load_unit_test
        ind2 = 2
        expected = unit_test_distances
    else:
        loader = load_basic_motions
        ind2 = 1
        expected = basic_motions_distances

    trainX, _ = loader(return_type="numpy3D")

    # test parameters
    if dist_str in distance_param_name:
        params_multi = distance_parameters[dist_str]
        param_list = [{distance_param_name[dist_str]: x} for x in params_multi]
    else:
        param_list = [{}]

    # assert distance between fixtures d, d2 are same as expected
    for j, param in enumerate(param_list):
        # deal with custom setting of epsilon in multi
        # this was in the original test before refactoring
        if "epsilon" in param and uni_multi == "multi":
            param = {"epsilon": param["epsilon"] / 50}
        # check that distance is same as expected
        d = dist(trainX[0], trainX[ind2], **param)
        d2 = dist(trainX[0], trainX[ind2], **param)
        assert_almost_equal(d, expected[dist_str][j], 4)
        assert d == d2
