# -*- coding: utf-8 -*-
import numpy as np
import timeit

from sktime.metrics.distances.tests.utils import _create_test_ts_distances
from sktime.metrics.distances.dtw._dtw import Dtw
from sktime.metrics.distances.dtw._lower_bouding import LowerBounding
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.metrics.distances._dtw_based import dtw as test_dtw
from sktime.metrics.distances.base.base import BaseDistance
from sktime.metrics.distances._squared_dist import SquaredDistance


def time_distance_call(
    base_dist_to_time,
    num_timeseries,
    timeseries_dimensions,
    timeseries_length,
    average_amount: int = 100,
):
    nested, _ = make_classification_problem(
        n_instances=num_timeseries,
        n_columns=timeseries_dimensions,
        n_timepoints=timeseries_length,
        n_classes=1,
        random_state=1,
    )
    numpy_ts = from_nested_to_3d_numpy(nested)
    x = numpy_ts[0]
    y = numpy_ts[1]

    def timeit_experiments():
        return base_dist_to_time.distance(x, y)

    def timeit_experiment_func():
        return base_dist_to_time(x, y)

    timeit_call = timeit_experiments
    if not isinstance(base_dist_to_time, BaseDistance):
        timeit_call = timeit_experiment_func
        distance_val = base_dist_to_time(x, y)
    else:
        distance_val = base_dist_to_time.distance(x, y)

    result = timeit.timeit(timeit_call, number=average_amount)

    return result / average_amount, distance_val


def time_pairwise_call(
    base_dist_to_time,
    num_timeseries,
    timeseries_dimensions,
    timeseries_length,
    average_amount: int = 100,
):
    nested, _ = make_classification_problem(
        n_instances=num_timeseries,
        n_columns=timeseries_dimensions,
        n_timepoints=timeseries_length,
        n_classes=1,
        random_state=1,
    )
    x = from_nested_to_3d_numpy(nested)

    nested, _ = make_classification_problem(
        n_instances=num_timeseries,
        n_columns=timeseries_dimensions,
        n_timepoints=timeseries_length,
        n_classes=1,
    )
    y = from_nested_to_3d_numpy(nested)

    def timeit_experiments():
        base_dist_to_time.pairwise(x, y)

    def timeit_experiment_func():
        base_dist_to_time(x, y)

    timeit_call = timeit_experiments
    if not isinstance(base_dist_to_time, BaseDistance):
        timeit_call = timeit_experiment_func

    result = timeit.timeit(timeit_call, number=average_amount)

    return result / average_amount


def test_dtw_time():
    timed_old, old_dist = time_distance_call(test_dtw, 2, 1000, 1000, 20)
    timed_class, new_dist = time_distance_call(
        Dtw(lower_bounding=1, custom_cost_matrix_distance=SquaredDistance()),
        2,
        1000,
        1000,
        20,
    )

    print("old", timed_old, "result:", old_dist)
    print("new", timed_class, "result:", new_dist)


def test_dtw_pairwise():
    num_timeseries = 10
    timeseries_dimensions = 100
    timeseries_length = 100

    nested, _ = make_classification_problem(
        n_instances=num_timeseries,
        n_columns=timeseries_dimensions,
        n_timepoints=timeseries_length,
        n_classes=1,
    )
    x = from_nested_to_3d_numpy(nested)

    nested, _ = make_classification_problem(
        n_instances=num_timeseries,
        n_columns=timeseries_dimensions,
        n_timepoints=timeseries_length,
        n_classes=1,
    )
    y = from_nested_to_3d_numpy(nested)

    test = Dtw(lower_bounding=1).pairwise(x)


def test_dtw_distance():
    x, y = _create_test_ts_distances([4, 4])
    test1 = Dtw(lower_bounding=1).distance(x, y)
    test2 = Dtw(
        lower_bounding=1, custom_cost_matrix_distance=SquaredDistance()
    ).distance(x, y)
    test3 = Dtw(lower_bounding=3).distance(x, y)


def test_lower_bounding():
    x, y = _create_test_ts_distances([10, 10])
    no_constraints = LowerBounding.NO_BOUNDING

    assert np.array_equal(
        no_constraints.create_bounding_matrix(x, y), np.zeros((x.shape[0], y.shape[0]))
    )

    sakoe_chiba = LowerBounding.SAKOE_CHIBA

    sakoe_chiba.create_bounding_matrix(x, y, sakoe_chiba_window_radius=10)

    itakura_parallelogram = LowerBounding.ITAKURA_PARALLELOGRAM
    itakura_parallelogram.create_bounding_matrix(x, y)
