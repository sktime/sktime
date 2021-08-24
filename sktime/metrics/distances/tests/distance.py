# -*- coding: utf-8 -*-
import pytest
from sktime.metrics.distances.dtw._dtw import Dtw

from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy

from sktime.metrics.distances._squared_dist import SquaredDistance
from sktime.metrics.distances._scipy_dist import ScipyDistance
from sktime.metrics.distances.dtw._fast_dtw import FastDtw
from sktime.metrics.distances.base.base import BaseDistance, BasePairwise

pytest_distance_parameters = [
    # (SquaredDistance()),
    # (Dtw()),
    # (ScipyDistance('euclidean')),
    (FastDtw()),
]

pytest_pairwise_parameters = [
    (SquaredDistance()),
    (Dtw()),
    (ScipyDistance("euclidean")),
]


def create_test_distance(n_instance, n_columns, n_timepoints, random_state=1):
    nested, _ = make_classification_problem(
        n_instances=n_instance,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        n_classes=1,
        random_state=random_state,
    )
    test = from_nested_to_3d_numpy(nested)
    return from_nested_to_3d_numpy(nested)


@pytest.mark.parametrize("distance", pytest_distance_parameters)
def test_univariate(distance: BaseDistance):
    generated_ts = create_test_distance(2, 10, 10)
    x = generated_ts[0]
    y = generated_ts[1]
    # Test single dimension array
    x_single_dimension = x[0]
    y_single_dimension = y[1]

    single_dim_result = distance.distance(x_single_dimension, y_single_dimension)

    # Test 2d array but with 1 item in the out array
    generated_ts = create_test_distance(2, 10, 1)
    x_two_dimension = generated_ts[0]
    y_two_dimension = generated_ts[1]

    two_dim_result = distance.distance(x_two_dimension, y_two_dimension)

    pass


@pytest.mark.parametrize("distance", pytest_distance_parameters)
def test_multivariate(distance: BaseDistance):
    # Test 2d array
    generated_ts = create_test_distance(2, 10, 10)
    x = generated_ts[0]
    y = generated_ts[1]

    multivariate_result = distance.distance(x, y)

    pass


@pytest.mark.parametrize("distance", pytest_pairwise_parameters)
def test_pairwise(distance: BasePairwise):
    # Test 2d array (assuming bunch of univariates)
    x_univariate_matrix = create_test_distance(10, 10, 1)
    y_univariate_matrix = create_test_distance(10, 10, 1, random_state=2)

    # Test 3d array (assuming bunch of multivariate)
    x = create_test_distance(10, 10, 10)
    y = create_test_distance(10, 10, 10, random_state=2)

    pairwise_result = distance.pairwise(x, y)

    pass
