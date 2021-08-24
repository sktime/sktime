# -*- coding: utf-8 -*-
import numpy as np
import timeit

from sktime.metrics.distances.tests.utils import _create_test_ts_distances
from sktime.metrics.distances.dtw._dtw import SquaredDistance


def test_squared_dist():
    x, y = _create_test_ts_distances([4, 4])
    test1 = SquaredDistance().distance(x, x)
    joe = ""
