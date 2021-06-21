# -*- coding: utf-8 -*-

__all__ = ["generate_univaritate_series", "run_clustering_experiment"]

import numpy as np
from sktime.clustering.base.base import BaseCluster
from sktime.clustering.base._typing import NumpyRandomState, NumpyArray


def generate_univaritate_series(
    n: int, size: int, rng: NumpyRandomState, dtype=np.float32
) -> NumpyArray:
    """
    Method to generate univariate time series
    """
    return rng.randn(n, size).astype(dtype)


def run_clustering_experiment(
    model: BaseCluster,
    X_train: NumpyArray,
    X_test: NumpyArray,
):
    """
    Method to run a clustering test.

    Parameters
    ----------
    model: BaseCluster
        Model to get clustering test results for

    X_train: NumpyArray
        Training dataset for model

    X_test: NumpyArray
        Test dataset for model
    """
    model.fit(X_train)
    clusters = model.predict(X_test)
    return clusters, model
