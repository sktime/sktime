# -*- coding: utf-8 -*-

__author__ = ["KatieBuc"]
__all__ = ["IntrinsDim"]

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sktime.base import BaseEstimator


class IntrinsDim(BaseEstimator):
    """
    metric: distance metric
    K : number of manifolds
    zeta : fixed zeta value
    q : number of points for local Z interaction
    """

    def __init__(
        self,
        metric="euclidean",
        K=2,
        zeta=0.8,
        q=3,
        optimization="MaxLikelihood",
        discard=0.2,
    ):
        self.metric = metric
        self.K = K
        self.zeta = zeta
        self.q = q
        self.optimization = optimization
        self.discard = discard

        super(IntrinsDim, self).__init__()

    def _find_neighbors(self, X):
        """."""
        self.N, self.d = np.shape(X)
        self._neighbors = NearestNeighbors(
            n_neighbors=3, algorithm="ball_tree", metric=self.metric
        )
        self._fitted_neighbors = self._neighbors.fit(X)
        self.distances, _ = self._fitted_neighbors.kneighbors(X)
        self.mu = np.sort(np.divide(self.distances[:, 2], self.distances[:, 1]))

        return self

    def _fit(self, X):
        """."""
        self._find_neighbors(X)
        if self.optimization == "MaxLikelihood":
            self.dim = float(self.N) / np.sum(np.log(self.mu))
        elif self.optimization == "LinearFit":
            F = np.arange(1, self.N + 1) / float(self.N)
            Neff = np.floor((1 - self.discard) * self.N).astype(int)
            _par = np.polyfit(np.log(self.mu[:Neff]), -np.log(1 - F[:Neff]), 1)
            self.dim = _par[0]

        return self

    # there's no y or fh, estimating the dimensionality of the dataset
    def fit(self, X):
        """."""
        self._fit(X)

        return self


# input is N x c array of exogenous variables
# output is scalar value, i.e. np.array([x])
