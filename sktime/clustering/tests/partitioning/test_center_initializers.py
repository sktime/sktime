# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering.partitioning._center_initializers import ForgyCenterInitializer


def test_random_cluster_center_initializer():
    n, sz = 100, 10
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz)
    ForgyCenterInitializer(time_series, 10)
