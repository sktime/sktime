# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
from numpy.random import RandomState
from tslearn.clustering import TimeSeriesKMeans

from sktime.clustering.base import BaseClusterer, TimeSeriesInstances
from sktime.utils.validation._dependencies import _check_soft_dependencies


class TslearnKmeans(BaseClusterer):

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 50,
        tol: float = 1e-6,
        n_init: int = 10,
        metric: str = "dtw",
        max_iter_barycenter: int = 100,
        distance_params: dict = None,
        n_jobs: int = None,
        verbose: bool = False,
        random_state=None,
        init_algorithm: str = "random",
    ):
        _check_soft_dependencies("tslearn", severity="error", object=self)

        self.init_algorithm = init_algorithm
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.n_jobs = n_jobs
        self.max_iter_barycenter = max_iter_barycenter

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._tslearn_k_means = None

        super(TslearnKmeans, self).__init__(n_clusters=n_clusters)

    def _fit(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

        verbose = 0
        if self.verbose is True:
            verbose = 1

        if self._tslearn_k_means is None:
            self._tslearn_k_means = TimeSeriesKMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                tol=self.tol,
                n_init=self.n_init,
                metric=self.metric,
                max_iter_barycenter=self.max_iter_barycenter,
                metric_params=self.distance_params,
                n_jobs=self.n_jobs,
                verbose=verbose,
                random_state=self.random_state,
                init=self.init_algorithm,
            )
        self._tslearn_k_means.fit(X)
        self.labels_ = self._tslearn_k_means.labels_
        self.inertia_ = self._tslearn_k_means.inertia_
        self.n_iter_ = self._tslearn_k_means.n_iter_

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
        return self._tslearn_k_means.predict(X)

    def _score(self, X, y=None):
        return np.abs(self.inertia_)
