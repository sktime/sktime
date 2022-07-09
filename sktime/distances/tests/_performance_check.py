# -*- coding: utf-8 -*-
import time
import warnings

import numpy as np
import pandas as pd

from sktime.distances.tests._utils import create_test_distance_numpy
from sktime.distances._msm import _MsmDistance

warnings.filterwarnings(
    "ignore"
)  # Hide warnings that can generate and clutter notebook


def timing_experiment(x, y, distance_callable, distance_params=None, average=1):
    """Time the average time it takes to take the distance from the first time series
    to all of the other time series in X.
    Parameters
    ----------
    X: np.ndarray
        A dataset of time series.
    distance_callable: Callable
        A callable that is the distance function to time.
    Returns
    -------
    float
        Average time it took to run a distance
    """
    if distance_params is None:
        distance_params = {}
    total_time = 0
    for i in range(0, average):
        start = time.time()
        curr_dist = distance_callable(x, y, **distance_params)
        total_time += time.time() - start

    test = np.sqrt(curr_dist)
    return total_time / average


def univariate_experiment(distance_metrics, start=1000, end=10000, increment=1000):
    """Runs an experiment on the univariate distance metrics."""
    timings = {"num_timepoints": []}
    x_distances = []
    y_distances = []
    for i in range(start, end + increment, increment):
        timings["num_timepoints"].append(i)
        distance_m_d = create_test_distance_numpy(2, 1, i)

        x = distance_m_d[0]
        y = distance_m_d[1]
        x_distances.append(x)
        y_distances.append(y)
        for dist_callable, name in distance_metrics:
            numba_callable = dist_callable(x, y)
            curr_timing = timing_experiment(x, y, numba_callable)
            if name not in timings:
                timings[name] = []
            timings[name].append(curr_timing)

    uni_df = pd.DataFrame(timings)
    uni_df = uni_df.set_index("num_timepoints")

    # import csv
    #
    # with open("output_x.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(x_distances)
    #
    # with open("output_y.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(y_distances)
    return uni_df



def multivariate_experiment(distance_metrics, start=1000, end=10000, increment=1000):
    """Runs an experiment on the multivariate distance metrics."""
    timings = {"num_timepoints": []}
    x_distances = []
    y_distances = []
    for i in range(start, end + increment, increment):
        timings["num_timepoints"].append(i)
        distance_m_d = create_test_distance_numpy(2, 100, i)

        x = distance_m_d[0]
        y = distance_m_d[1]
        x_distances.append(x)
        y_distances.append(y)
        for dist_callable, name in distance_metrics:
            numba_callable = dist_callable(x, y)
            curr_timing = timing_experiment(x, y, numba_callable)
            if name not in timings:
                timings[name] = []
            timings[name].append(curr_timing)

    multi_df = pd.DataFrame(timings)
    multi_df = multi_df.set_index("num_timepoints")

    # import csv
    #
    # with open("output_x.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(x_distances)
    #
    # with open("output_y.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(y_distances)
    return multi_df



if __name__ == "__main__":
    metrics = [
        (_MsmDistance().distance_factory, "msm new"),
        # (OldMsm().distance_factory, "msm old"),
    ]
    uni_df = univariate_experiment(
        distance_metrics=metrics, start=1000, end=10000, increment=1000
    )
    multi_df = multivariate_experiment(
        distance_metrics=metrics,
        start=1000,
        end=2000,
        increment=1000
    )