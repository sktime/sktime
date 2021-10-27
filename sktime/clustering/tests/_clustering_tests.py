# -*- coding: utf-8 -*-

__all__ = ["generate_univaritate_series", "run_clustering_experiment"]

import numpy as np
from sktime.clustering.base.base import BaseClusterer
from sktime.clustering.base._typing import NumpyRandomState, NumpyArray
from sklearn.utils import check_random_state


def generate_univaritate_series(
    n: int, size: int, rng: NumpyRandomState, dtype=np.double
) -> NumpyArray:
    """
    Method to generate univariate time series
    """
    rng = check_random_state(rng)
    if dtype is np.int32 or dtype is np.int64:
        return rng.randint(0, 1000, size=(n, size)).astype(dtype)
    return rng.randn(n, size).astype(dtype)


def run_clustering_experiment(
    model: BaseClusterer,
    X_train: NumpyArray,
    X_test: NumpyArray,
):
    """
    Method to run a clustering test.

    Parameters
    ----------
    model: BaseClusterer
        Model to get clustering test results for

    X_train: NumpyArray
        Training dataset for model

    X_test: NumpyArray
        Test dataset for model
    """
    model.fit(X_train)
    clusters = model.predict(X_test)
    return clusters, model
