# -*- coding: utf-8 -*-
import math

import numpy as np

from sktime.metrics.distances.tests.distance import create_test_distance
from sktime.metrics.distances.dtw._fast_dtw import FastDtw, _reduce_by_half


def test_reduce_by_half():
    generated_ts = create_test_distance(2, 10, 10)
    x = generated_ts[0]
    y = generated_ts[1]
    test = _reduce_by_half(x)
    pass


def _reduce_by_half_test(x):
    return [(x[i] + x[1 + i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]


def test_there():
    x = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    test = _reduce_by_half(x)

    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    test_2 = _reduce_by_half_test(x)

    x_test = np.array([[2, 2, 1, 1, 5, 5], [1, 1, 1, 1, 1, 1]])
    x_size = x_test.shape[0]

    test = _reduce_by_half(x_test)

    print("++++++++\n")
