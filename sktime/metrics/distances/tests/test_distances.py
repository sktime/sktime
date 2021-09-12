# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Callable
import numpy as np
import pandas as pd

from sktime.utils._testing.panel import make_classification_problem, _make_panel_X
from sktime.datatypes import convert_to
from sktime.utils._testing.series import _make_series
from sktime.metrics.distances.distance import (
    distance,
    pairwise,
    get_available_distances,
)
from sktime.metrics.distances.base.base import BaseDistance


def create_test_distance_df(
    n_instances,
    n_columns,
    n_timepoints,
    random_state=1,
):
    if n_instances > 1:
        return _make_panel_X(
            n_instances=n_instances,
            n_columns=n_columns,
            n_timepoints=n_timepoints,
            random_state=random_state,
        )
    else:
        return _make_series(n_timepoints, n_columns, random_state=random_state)


def create_test_distance_numpy(
    n_instances,
    n_columns,
    n_timepoints,
    random_state=1,
):
    df = create_test_distance_df(
        n_instances=n_instances,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        random_state=random_state,
    )
    if n_instances > 1:
        return convert_to(df, to_type="numpy3D")
    else:
        return convert_to(df, to_type="np.ndarray")


def create_test_distance_list(
    n_instance,
    n_columns,
    n_timepoints,
    random_state=1,
):
    numpy = create_test_distance_numpy(
        n_instances=n_instance,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        random_state=random_state,
    )
    return numpy.tolist()


class AbsoluteDistance(BaseDistance):
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        abs_distance = 0.0

        for i in range(x.shape[0]):
            abs_distance += np.sum(np.abs(x[i] - y[i])[0])

        return abs_distance


def absolute_distance_callable(x: np.ndarray, y: np.ndarray) -> float:
    abs_distance = 0.0

    for i in range(x.shape[0]):
        abs_distance += np.sum(np.abs(x[i] - y[i]))

    return abs_distance


def test_distances_and_pairwise():
    ts_create_arr = [
        create_test_distance_numpy,
        create_test_distance_df,
        create_test_distance_list,
    ]
    for create_time_series in ts_create_arr:
        distance_alternative_metric_test(
            absolute_distance_callable, AbsoluteDistance(), create_time_series
        )
        # pairwise_alternative_metric_test(
        #     absolute_distance_callable, AbsoluteDistance(), create_time_series)
        # for metric_str in get_available_distances():
        #     univariate_test(metric_str, create_time_series)
        #     multivariate_test(metric_str, create_time_series)
        #     pairwise_distance_test(metric_str, create_time_series)


def univariate_test(metric_str: str, create_time_series: Callable):
    # Method used to test a univariate time series
    generated_univariate_ts = create_time_series(2, 10, 1)
    x_univariate = create_time_series(
        1,
        1,
        10,
    )
    y_univariate = generated_univariate_ts[1]

    generated_ts = create_time_series(2, 10, 10, random_state=5)

    distance(x_univariate, y_univariate, metric_str)

    x_1d_univariate = generated_ts[0][0]
    y_1d_univariate = generated_ts[0][1]
    distance(x_1d_univariate, y_1d_univariate, metric_str)


def multivariate_test(metric_str: str, create_time_series: Callable):
    # Method used to test multivariate time series
    generated_ts = create_time_series(2, 10, 10, random_state=5)
    x_multivariate = generated_ts[0]
    y_multivariate = generated_ts[1]

    distance(x_multivariate, y_multivariate, metric_str)


def pairwise_distance_test(metric_str: str, create_time_series: Callable):
    # Method used to test the pairwise distances
    x_univariate_matrix = create_time_series(10, 10, 1)
    y_univariate_matrix = create_time_series(10, 10, 1, random_state=2)

    pairwise(x_univariate_matrix, y_univariate_matrix, metric_str)

    # Test 3d array (assuming bunch of multivariate)
    x_multivariate_matrix = create_time_series(10, 10, 10)
    y_multivariate_matrix = create_time_series(10, 10, 10, random_state=2)

    pairwise(x_multivariate_matrix, y_multivariate_matrix, metric_str)


def distance_alternative_metric_test(
    callable: Callable, base_dist: BaseDistance, create_time_series: Callable
):
    # Method used to test using BaseDistance or Callable in the distance function
    # call

    x_univariate = create_time_series(1, 1, 10, random_state=1)
    y_univariate = create_time_series(1, 1, 10, random_state=2)

    distance(x_univariate, y_univariate, callable)
    distance(x_univariate, y_univariate, base_dist)

    generated_univariate_ts = create_time_series(2, 10, 1)

    x_1d_univariate = generated_univariate_ts[0][0]
    y_1d_univariate = generated_univariate_ts[0][1]

    distance(x_1d_univariate, y_1d_univariate, callable)
    distance(x_1d_univariate, y_1d_univariate, base_dist)

    generated_ts = create_time_series(2, 10, 10, random_state=5)
    x_multivariate = generated_ts[0]
    y_multivariate = generated_ts[1]

    distance(x_multivariate, y_multivariate, callable)
    distance(x_multivariate, y_multivariate, base_dist)


def pairwise_alternative_metric_test(
    callable: Callable, base_dist: BaseDistance, create_time_series: Callable
):
    # Method used to test BaseDistance or Callable in the pairwise function
    # call
    x_univariate_matrix = create_time_series(10, 10, 1)
    y_univariate_matrix = create_time_series(10, 10, 1, random_state=2)

    pairwise(x_univariate_matrix, y_univariate_matrix, callable)
    pairwise(x_univariate_matrix, y_univariate_matrix, base_dist)

    x_multivariate_matrix = create_time_series(10, 10, 10)
    y_multivariate_matrix = create_time_series(10, 10, 10, random_state=2)

    pairwise(x_multivariate_matrix, y_multivariate_matrix, callable)
    pairwise(x_multivariate_matrix, y_multivariate_matrix, base_dist)


# def test_runtime():
#     generated_ts = create_test_distance(100, 10, 10, random_state=5)
#
#     x = generated_ts
#     y = generated_ts
#
#     import cProfile
#     import pstats
#
#     test = FastDtw(radius=0)
#
#     with cProfile.Profile() as pr:
#         test.pairwise(x, y)
#
#     stats = pstats.Stats(pr)
#     stats.sort_stats(pstats.SortKey.TIME)
#     stats.print_stats()
