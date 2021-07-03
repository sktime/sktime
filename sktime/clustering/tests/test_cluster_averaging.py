# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering.partitioning._averaging_metrics import (
    BarycenterAveraging,
    MeanAveraging,
)
from sktime.clustering.tests._clustering_tests import generate_univaritate_series


def test_barycenter_averaging():
    rng = np.random.RandomState(0)
    X = generate_univaritate_series(n=100, size=5, rng=rng, dtype=np.int)

    BCA = BarycenterAveraging(X)
    BCA.average()


def test_mean_averaging():
    rng = np.random.RandomState(1)
    X = generate_univaritate_series(n=100, size=5, rng=rng, dtype=np.int)

    mean = MeanAveraging(X)
    average = mean.average()
    assert np.array_equal(np.array([-0.02, 0.01, 0.05, 0.11, 0.03]), average)
