# -*- coding: utf-8 -*-
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.benchmarking._internal_benchmarks.profiling import time_function_call

from sktime.metrics.distances._squared_dist import SquaredDistance
from sktime.metrics.distances.dtw._dtw import Dtw
from sktime.distances.elastic_cython import dtw_distance

from tslearn.metrics.dtw_variants import dtw as tslearn_dtw


def run_distance_benchmark(
    num_timeseries, timeseries_length, timeseries_dimensions, distance_func, **kwargs
):
    nested, _ = make_classification_problem(
        n_instances=num_timeseries,
        n_columns=timeseries_dimensions,
        n_timepoints=timeseries_length,
        n_classes=1,
    )
    numpy_ts = from_nested_to_3d_numpy(nested)
    x = numpy_ts[0]
    y = numpy_ts[1]

    kwargs["x"] = x
    kwargs["y"] = y

    # profile = profile_a_function(
    #     distance_func,
    #     print_stats=True,
    #     kwargs=kwargs
    # )

    time_taken = time_function_call(
        function_to_time=distance_func, average_amount=100, kwargs=kwargs
    )
    # time_function_call(function_to_time=tslearn_dtw, average_amount=50, kwargs=kwargs)

    return time_taken


def benchmark_dtws():
    tslearn = run_distance_benchmark(
        num_timeseries=2,
        timeseries_length=1000,
        timeseries_dimensions=1000,
        distance_func=tslearn_dtw,
    )
    ours = run_distance_benchmark(
        num_timeseries=2,
        timeseries_length=1000,
        timeseries_dimensions=1000,
        distance_func=Dtw().distance,
    )
    cython_currnent = run_distance_benchmark(
        num_timeseries=2,
        timeseries_length=1000,
        timeseries_dimensions=1000,
        distance_func=dtw_distance,
    )
    print("tslearn", tslearn)
    print("ours", ours)
    print("cython current", cython_currnent)
    pass


# def benchmakr_tslearn():
#     no_bounding = run_distance_benchmark(
#         num_timeseries=2,
#         timeseries_length=1000,
#         timeseries_dimensions=1000,
#         distance_func=tslearn_dtw,
#     )
#     sakoe_bounding = run_distance_benchmark(
#         num_timeseries=2,
#         timeseries_length=1000,
#         timeseries_dimensions=1000,
#         distance_func=tslearn_dtw,
#         global_constraint="sakoe_chiba",
#     )
#     itakura_bounding = run_distance_benchmark(
#         num_timeseries=2,
#         timeseries_length=1000,
#         timeseries_dimensions=1000,
#         distance_func=tslearn_dtw,
#         global_constraint="itakura",
#     )
#     print("ts learn no bounding", no_bounding)
#     print("ts learn sakoe", sakoe_bounding)
#     print("ts leran itakura", itakura_bounding)
#     pass


def benchmark_distance():
    # no_bounding = run_distance_benchmark(
    #     num_timeseries=2,
    #     timeseries_length=1000,
    #     timeseries_dimensions=1000,
    #     distance_func=dtw,
    #     lower_bounding=1,
    # )
    # sakoe_bounding = run_distance_benchmark(
    #     num_timeseries=2,
    #     timeseries_length=1000,
    #     timeseries_dimensions=1000,
    #     distance_func=dtw,
    #     lower_bounding=2,
    # )
    # itakura_bounding = run_distance_benchmark(
    #     num_timeseries=2,
    #     timeseries_length=1000,
    #     timeseries_dimensions=1000,
    #     distance_func=dtw,
    #     lower_bounding=3,
    # )
    # print("no bounding", no_bounding)
    # print("sakoe", sakoe_bounding)
    # print("itakura", itakura_bounding)
    pass


def benchmark_bounding():
    # non = LowerBounding.NO_BOUNDING
    # no_bounding = run_distance_benchmark(
    #     num_timeseries=2,
    #     timeseries_length=1000,
    #     timeseries_dimensions=1000,
    #     distance_func=non.create_bounding_matrix,
    # )
    #
    # sakoe_chiba = LowerBounding.SAKOE_CHIBA
    # sakoe_chiba_bounding = run_distance_benchmark(
    #     num_timeseries=2,
    #     timeseries_length=1000,
    #     timeseries_dimensions=1000,
    #     distance_func=sakoe_chiba.create_bounding_matrix,
    #     kwargs={"sakoe_chiba_window_radius": 2},
    # )
    #
    # itakura_parallelogram = LowerBounding.ITAKURA_PARALLELOGRAM
    # itakura_parallelogram_bounding = run_distance_benchmark(
    #     num_timeseries=2,
    #     timeseries_length=1000,
    #     timeseries_dimensions=1000,
    #     distance_func=itakura_parallelogram.create_bounding_matrix,
    #     kwargs={"itakura_max_slope": 2},
    # )
    # print("no bounding", no_bounding)
    # print("sakoe chiba bounding", sakoe_chiba_bounding)
    # print("itakura parallelogram bounding", itakura_parallelogram_bounding)
    pass


if __name__ == "__main__":
    # benchmark_bounding()
    benchmark_dtws()
