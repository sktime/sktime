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
#
# import numpy as np
#
# from sktime.distances import edr_distance as dist
# from sktime.distances.tests._utils import create_test_distance_numpy
#
#
# def test_dist_result(dist):
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
#     first = dist(x_first, y_first)
#     second = dist(x_second, y_second)
#
#     third = dist(x_third, y_third)
#     fourth = dist(x_fourth, y_fourth)
#
#     test = dist(x_first, x_first)
#     tst_1 = dist(x_second, x_second)
#     tst_2 = dist(x_third, x_third)
#     tst_3 = dist(x_fourth, x_fourth)
#
#     expected = [round(first, 5), round(second, 5), round(fourth, 5)]
#
#     pass
#
# def test_create_examples():
#     import numpy as np
#     from sktime.distances import distance as dist
#     metric = 'wddtw'
#
#     x_1d = np.array([1, 2, 3, 4])  # 1d array
#     y_1d = np.array([5, 6, 7, 8])  # 1d array
#     result_1 = dist(x_1d, y_1d, metric=metric)
#
#     x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
#     y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
#     result_2 = dist(x_2d, y_2d, metric=metric)
#     pass
#
#
# if __name__ == "__main__":
#     test_create_examples()
