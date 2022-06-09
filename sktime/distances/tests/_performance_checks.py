import time
import warnings

from sktime.distances import distance_factory
from sktime.distances.tests._utils import create_test_distance_numpy

import pandas as pd

warnings.filterwarnings(
    'ignore')  # Hide warnings that can generate and clutter notebook


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

    return total_time / average


def univariate_experiment(distance_metrics, start=1000, end=10000, increment=1000):
    timings = {
        'num_timepoints': []
    }
    for i in range(start, end + increment, increment):
        timings['num_timepoints'].append(i)
        distance_m_d = create_test_distance_numpy(2, 1, i)

        x = distance_m_d[0]
        y = distance_m_d[1]
        for dist_callable, name in distance_metrics:
            numba_callable = dist_callable(x, y)
            curr_timing = timing_experiment(x, y, numba_callable)
            if name not in timings:
                timings[name] = []
            timings[name].append(curr_timing)

    uni_df = pd.DataFrame(timings)
    uni_df = uni_df.set_index('num_timepoints')
    return uni_df


def multivariate_experiment(start=100, end=500, increment=100):
    sktime_timing = []

    col_headers = []

    for i in range(start, end + increment, increment):
        col_headers.append(i)
        distance_m_d = create_test_distance_numpy(2, i, i)

        x = distance_m_d[0]
        y = distance_m_d[1]
        numba_sktime = distance_factory(x, y, metric='dtw')

        sktime_time = timing_experiment(x, y, numba_sktime)

        sktime_timing.append(sktime_time)

    multi_df = pd.DataFrame({
        'time points': col_headers,
        'sktime': sktime_timing,
    })
    return multi_df


from sktime.distances._msm import _MsmDistance
from sktime.distances.tests.old_msm import _MsmDistance as OldMsm
if __name__ == '__main__':
    metrics = [
        (_MsmDistance().distance_factory, 'msm new'),
        (OldMsm().distance_factory, 'msm old')
    ]
    uni_df = univariate_experiment(
        distance_metrics=metrics,
        start=100,
        end=100,
        increment=100
    )

    joe = ''
    # uni_df.to_csv('./uni_dist_results', index=False)
    # multi_df = multivariate_experiment(
    #     start=100,
    #     end=200,
    #     increment=100
    # )
    # multi_df.to_csv('./multi_dist_results', index=False)

# @njit(cache=True)
# def _cost_function(x: np.ndarray, y: np.ndarray, z: np.ndarray, c: float) -> float:
#     """Compute cost function for msm.
#
#     Parameters
#     ----------
#     x: np.ndarray
#         First point.
#     y: np.ndarray
#         Second point.
#     z: np.ndarray
#         Third point.
#     c: float, default = 1.0
#         Cost for split or merge operation.
#
#     Returns
#     -------
#     float
#         The msm cost between points.
#     """
#     diameter = _local_euclidean(y, z)
#
#     # sum = np.zeros_like(x)
#     add = np.add(y, z)
#     # for i in range(len(y)):
#     #     sum[i] = y[i] + z[i]
#     # mid = sum / 2
#     mid = add
#     distance_to_mid = _local_euclidean(mid, x)
#
#     if distance_to_mid <= (diameter / 2):
#         return c
#     else:
#         dist_to_q_prev = _local_euclidean(y, x)
#         dist_to_c = _local_euclidean(z, x)
#         if dist_to_q_prev < dist_to_c:
#             return c + dist_to_q_prev
#         else:
#             return c + dist_to_c