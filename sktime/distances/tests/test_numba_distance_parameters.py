# -*- coding: utf-8 -*-
"""Test suite for numba distances with parameters."""
from typing import Callable, Dict, List

import numpy as np

from sktime.distances import distance, distance_factory, dtw_distance
from sktime.distances.tests._utils import create_test_distance_numpy


# TODO: Change defaults in distance for itakura and sakoe
def _test_distance_params(
    param_list: List[Dict], distance_func: Callable, distance_str: str
):
    x_univ = create_test_distance_numpy(10, 1)
    y_univ = create_test_distance_numpy(10, 1, random_state=2)

    x_multi = create_test_distance_numpy(10, 10)
    y_multi = create_test_distance_numpy(10, 10, random_state=2)

    test_ts = [[x_univ, y_univ], [x_multi, y_multi]]

    results_to_fill = []

    i = 0
    for param_dict in param_list:
        j = 0
        curr_results = []
        for x, y in test_ts:
            results = []
            dist_func = distance_factory(x, y, metric=distance_str, **param_dict)
            results.append(distance_func(x, y, **param_dict))
            results.append(distance(x, y, metric=distance_str, **param_dict))
            results.append(dist_func(x, y))
            results.append(dist_func(x, y, **param_dict))

            # if _expected_distance_results_params[distance_str][i][j] is not None:
            #     for result in results:
            #         assert result == pytest.approx(
            #             _expected_distance_results_params[distance_str][i][j]
            #         )
            curr_results.append(results[0])
            j += 1
        i += 1
        results_to_fill.append(curr_results)


def test_distance_params():
    """Test distance params."""
    _test_distance_params(
        [
            {"window": 0.2},
            {"itakura_max_slope": 0.5},
            {"bounding_matrix": np.zeros((10, 10))},
        ],
        dtw_distance,
        "dtw",
    )
