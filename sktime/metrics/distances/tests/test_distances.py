# -*- coding: utf-8 -*-
import pytest
from sktime.metrics.distances.dtw._dtw import Dtw

from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy

from sktime.metrics.distances.distance import (
    distance,
    pairwise,
    get_available_distances,
)
from sktime.metrics.distances._squared_dist import SquaredDistance
from sktime.metrics.distances._scipy_dist import ScipyDistance
from sktime.metrics.distances.dtw._dtw_cost_matrix import DtwCostMatrix
from sktime.metrics.distances.dtw._dtw_path import DtwPath
from sktime.metrics.distances.dtw._fast_dtw import FastDtw
from sktime.metrics.distances.base.base import BaseDistance

pytest_distance_parameters = [
    (SquaredDistance()),
    (Dtw()),
    (ScipyDistance("euclidean")),
    (DtwCostMatrix()),
    (DtwPath()),
    (FastDtw()),
]


def create_test_distance(n_instance, n_columns, n_timepoints, random_state=1):
    nested, _ = make_classification_problem(
        n_instances=n_instance,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        n_classes=1,
        random_state=random_state,
    )
    return from_nested_to_3d_numpy(nested)


def test_distance():
    generated_ts = create_test_distance(100, 10, 10, random_state=5)

    x = generated_ts
    y = generated_ts

    for metric_str in get_available_distances():
        distance(x[0], y[0], metric_str)


def test_pairwise():
    generated_ts = create_test_distance(20, 10, 10, random_state=5)

    x = generated_ts
    y = generated_ts

    for metric_str in get_available_distances():
        pairwise(x, y, metric_str)


@pytest.mark.parametrize("distance", pytest_distance_parameters)
def test_univariate(distance: BaseDistance):
    generated_ts = create_test_distance(2, 10, 10)
    x = generated_ts[0]
    y = generated_ts[1]
    # Test single dimension array
    x_single_dimension = x[0]
    y_single_dimension = y[1]

    distance.distance(x_single_dimension, y_single_dimension)

    # Test 2d array but with 1 item in the out array
    generated_ts = create_test_distance(2, 10, 1)
    x_two_dimension = generated_ts[0]
    y_two_dimension = generated_ts[1]

    distance.distance(x_two_dimension, y_two_dimension)


@pytest.mark.parametrize("distance", pytest_distance_parameters)
def test_multivariate(distance: BaseDistance):
    # Test 2d array
    generated_ts = create_test_distance(2, 30, 15, random_state=5)
    x = generated_ts[0]
    y = generated_ts[1]

    distance.distance(x, y)


@pytest.mark.parametrize("distance", pytest_distance_parameters)
def test_pairwise_distance(distance: BaseDistance):
    # Test 2d array (assuming bunch of univariates)
    x_univariate_matrix = create_test_distance(10, 10, 1)
    y_univariate_matrix = create_test_distance(10, 10, 1, random_state=2)

    distance.pairwise(x_univariate_matrix, y_univariate_matrix)

    # Test 3d array (assuming bunch of multivariate)
    x = create_test_distance(20, 22, 35)
    y = create_test_distance(20, 22, 35, random_state=2)

    distance.pairwise(x, y)


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
