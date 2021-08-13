# -*- coding: utf-8 -*-
from sktime.metrics.distances.distances import dtw
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.benchmarking._internal_benchmarks.profiling import time_function_call

# from tslearn.metrics.dtw_variants import dtw as tslearn_dtw


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
        function_to_time=distance_func, average_amount=50, kwargs=kwargs
    )
    # time_function_call(function_to_time=tslearn_dtw, average_amount=50, kwargs=kwargs)

    return time_taken


if __name__ == "__main__":
    run_distance_benchmark(
        num_timeseries=20,
        timeseries_length=1000,
        timeseries_dimensions=1000,
        distance_func=dtw,
    )
