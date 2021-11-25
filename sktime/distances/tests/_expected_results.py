# -*- coding: utf-8 -*-
# The key string (i.e. 'euclidean') must be the same as the name in _registry

_expected_distance_results = {
    # Result structure:
    # [single value series, univariate series, multivariate series, multivariate panel]
    "squared": [25.0, 6.93261, 50.31911],
    "euclidean": [5.0, 2.63298, 7.09359],
    "dtw": [25.0, 2.18037, 46.68062],
    "ddtw": [0.0, 2.08848, 31.57625],
    "wdtw": [12.5, 1.09018, 23.34031],
    "wddtw": [0.0, 1.04424, 15.78813],
    "lcss": [1.0, 0.1, 1.0],
    "edr": [1.0, 0.6, 1.0],
    "erp": [5.0, 5.03767, 20.78718],
}

# def test_dist_result(dist):
#     import numpy as np
#
#     from sktime.distances.tests._utils import create_test_distance_numpy
#
#     x_first = np.array([10.0])
#     y_first = np.array([15.0])
#
#     x_second = create_test_distance_numpy(10)
#     y_second = create_test_distance_numpy(10, random_state=2)
#
#     x_third = create_test_distance_numpy(10, 1)
#     y_third = create_test_distance_numpy(10, 1, random_state=2)
#
#     x_fourth = create_test_distance_numpy(10, 10)
#     y_fourth = create_test_distance_numpy(10, 10, random_state=2)
#
#     x_trunc = np.reshape(x_third, (x_third.shape[1], x_third.shape[0]))
#     y_trunc = np.reshape(y_third, (y_third.shape[1], y_third.shape[0]))
#
#     first = dist(x_first, y_first)
#     second = dist(x_second, y_second)
#     third = dist(x_third, y_third)
#     fourth = dist(x_fourth, y_fourth)
#     trunc = dist(x_trunc, y_trunc)
#
#     result_3 = dist(x_third, x_third, window=2)
#     result_4 = dist(x_third, x_third, itakura_max_slope=2.)
#
#     expected = [round(first, 5), round(second, 5), round(fourth, 5)]
#
#     pass
#
# def test_create_examples():
#     import numpy as np
#     from sktime.distances import distance as dist
#     metric = 'edr'
#
#     x_1d = np.array([1, 2, 3, 4])  # 1d array
#     y_1d = np.array([5, 6, 7, 8])  # 1d array
#     result_1 = dist(x_1d, y_1d, metric=metric)
#
#     x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
#     y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
#     result_2 = dist(x_2d, y_2d, metric=metric)
#
#     pass
#
# def time_experiment(x, y, dist, iters, **kwargs):
#     total = 0
#
#     # Run 3 times to ensure cache'd
#     for i in range(3):
#         dist(x, y, **kwargs)
#
#     for i in range(iters):
#         start = time.time()
#         dist(x, y, **kwargs)
#         total += time.time() - start
#
#     return total/iters
#
#
#
# def test_windows():
#     """Roadtest the new distances."""
#     import numpy as np
#     import time
#     from sktime.distances import euclidean_distance, dtw_distance
#     from sktime.distances import LowerBounding
#     from sktime.distances._numba_utils import to_numba_timeseries
#     from sktime.distances.tests._utils import create_test_distance_numpy
#
#     # itakura = LowerBounding.ITAKURA_PARALLELOGRAM
#     # itakura_bounding = itakura.create_bounding_matrix(_x, _y, itakura_max_slope=2.)
#
#     x = np.zeros(100)
#     x[1] = 10
#     y = np.roll(x, 50)
#     # x = create_test_distance_numpy(100, 1000)
#     # y = create_test_distance_numpy(100, 1000, random_state=2)
#
#     _x = to_numba_timeseries(x)
#     _y = to_numba_timeseries(y)
#
#     sakoe_chiba = LowerBounding.SAKOE_CHIBA
#     sakoe_chiba_bounding_0 = sakoe_chiba.create_bounding_matrix(
#         _x,
#         _y,
#         sakoe_chiba_window_radius=0
#     )
#     sakoe_chiba_bounding_10 = sakoe_chiba.create_bounding_matrix(
#         _x,
#         _y,
#         sakoe_chiba_window_radius=10
#     )
#
#     time_euclidean = time_experiment(
#         x,
#         y,
#         euclidean_distance,
#         50
#     )
#
#     time_dtw_no_window = time_experiment(
#         x,
#         y,
#         dtw_distance,
#         50
#     )
#
#     time_dtw_0_window = time_experiment(
#         x,
#         y,
#         dtw_distance,
#         50,
#         bounding_matrix = sakoe_chiba_bounding_0
#     )
#
#     time_dtw_10_window = time_experiment(
#         x,
#         y,
#         dtw_distance,
#         50,
#         bounding_matrix=sakoe_chiba_bounding_10
#     )
#
#     print(
#         "Euclidean Distance = ",
#         euclidean_distance(x, y),
#         " takes = ",
#         time_euclidean
#         )
#
#     print("Full DTW Distance = ",
#           dtw_distance(x, y),
#           " takes = ",
#           time_dtw_no_window
#           )
#
#     print("Zero window DTW Distance = ",
#           dtw_distance(x, y, bounding_matrix=sakoe_chiba_bounding_0),
#           " takes = ",
#           time_dtw_0_window
#           )
#
#     print("Too Small Window DTW Distance = ",
#           dtw_distance(x, y, bounding_matrix=sakoe_chiba_bounding_10),
#           " takes = ",
#           time_dtw_10_window
#           )
#
#
# if __name__ == "__main__":
#     # from sktime.distances import dtw_distance as dist
#     # test_dist_result(dist)
#     test_windows()
