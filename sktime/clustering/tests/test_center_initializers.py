# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering.tests._clustering_tests import generate_univaritate_series
from sktime.clustering.partitioning._center_initializers import ForgyCenterInitializer


def test_forgy_cluster_center_initializer():
    rng = np.random.RandomState(0)
    X = generate_univaritate_series(n=20, size=1, rng=rng, dtype=np.float32)
    forgy_centers = ForgyCenterInitializer(X, 5, rng)
    centers = forgy_centers.initialize_centers()
    assert np.array_equal(
        np.array(
            [[0.33367434], [1.4940791], [0.95008844], [0.4001572], [-0.10321885]],
            dtype=np.float32,
        ),
        centers,
    )
