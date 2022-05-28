import time
import warnings

from sktime.distances import distance_factory
from sktime.distances.tests._utils import create_test_distance_numpy

import pandas as pd

warnings.filterwarnings(
    'ignore')  # Hide warnings that can generate and clutter notebook


def timing_experiment(x, y, distance_callable, distance_params=None, average=5):
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

    return total_time / average


def univariate_experiment(start=1000, end=10000, increment=1000, metric='dtw'):
    sktime_timing = []
    col_headers = []

    for i in range(start, end + increment, increment):
        col_headers.append(i)
        distance_m_d = create_test_distance_numpy(2, 1, i)

        x = distance_m_d[0][0]
        y = distance_m_d[1][0]
        numba_sktime = distance_factory(x, y, metric=metric)
        sktime_time = timing_experiment(distance_m_d[0], distance_m_d[1], numba_sktime)
        print(f"univariate size {i}: {sktime_time}")

        sktime_timing.append(sktime_time)

    uni_df = pd.DataFrame({
        'time points': col_headers,
        'sktime': sktime_timing,
    })
    return uni_df


def multivariate_experiment(start=100, end=500, increment=100, metric='dtw'):
    sktime_timing = []
    tslearn_timing = []

    col_headers = []

    for i in range(start, end + increment, increment):
        col_headers.append(i)
        distance_m_d = create_test_distance_numpy(2, i, i)

        x = distance_m_d[0]
        y = distance_m_d[1]
        numba_sktime = distance_factory(x, y, metric=metric)

        sktime_time = timing_experiment(x, y, numba_sktime)
        print(f"multivariate size {i}: {sktime_time}")

        sktime_timing.append(sktime_time)

    multi_df = pd.DataFrame({
        'time points': col_headers,
        'sktime': sktime_timing,
    })
    return multi_df


if __name__ == '__main__':
    uni_df = univariate_experiment(
        start=1000,
        end=10000,
        increment=1000,
        metric='msm'
    )
    uni_df.to_csv('./uni_dist_results.csv', index=False)
    # print(uni_df)

    # multi_df = multivariate_experiment(
    #     start=1000,
    #     end=10000,
    #     increment=1000
    # )
    # # print(multi_df)
    # multi_df.to_csv('./multi_dist_results', index=False)