# -*- coding: utf-8 -*-
import numpy as np

from sktime.distances.distance_rework import (
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


def example_not_equal(x, y):
    dist_objs = [
        [_DtwDistance(), "dtw"],
        [_DdtwDistance(), "ddtw"],
        [_WdtwDistance(), "wdtw"],
        [_WddtwDistance(), "wddtw"],
        [_EdrDistance(), "edr"],
        [_ErpDistance(), "erp"],
        [_EuclideanDistance(), "euclidean"],
        [_LcssDistance(), "lcss"],
        [_MsmDistance(), "msm"],
        [_SquaredDistance(), "squared"],
        [_TweDistance(), "twe"],
    ]
    for dist in dist_objs:
        ind = dist[0].distance(x, y, strategy="independent")
        dep = dist[0].distance(x, y, strategy="dependent")
        print(f"{dist[1]}: {ind} {dep}")
        print("are equal = ", ind == dep)


if __name__ == "__main__":
    x_2d = np.array(
        [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
    )
    y_2d = np.array(
        [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
    )

    x_1d = np.array([2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13])
    y_1d = np.array([5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7])

    dtw_dist = _DtwDistance()

    # Calling independent
    result_1d_ind = dtw_dist.distance(x_1d, y_1d, strategy="independent")
    result_2d_ind = dtw_dist.distance(x_2d, y_2d, strategy="independent")
    # Can also call it like
    result_1d_second_ind = dtw_dist.independent_distance(x_1d, y_1d)
    result_2d_second_ind = dtw_dist.independent_distance(x_2d, y_2d)

    # Calling dependent
    result_1d_dep = dtw_dist.distance(x_1d, y_1d, strategy="dependent")
    result_2d_dep = dtw_dist.distance(x_2d, y_2d, strategy="dependent")
    # Can also call it like
    result_1d_second_dep = dtw_dist.dependent_distance(x_1d, y_1d)
    result_2d_second_dep = dtw_dist.dependent_distance(x_2d, y_2d)

    example_not_equal(x_1d, y_1d)
