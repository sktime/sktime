# -*- coding: utf-8 -*-

__author__ = ["KatieBuc"]
__all__ = ["Hidalgo"]

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sktime.base import BaseEstimator

# import _gibbs


class Hidalgo(BaseEstimator):
    """
    metric: distance metric
    K : number of manifolds, free variable found via model fitting procedure
    zeta : fixed zeta value, free variable found via ...
    q : number of points for local Z interaction, free variable found via ...
    n_iter : number of iterations of MCMC algorithm
    n_random_starts : numer of random starts of MCMC algorithm
    burn_in : percentage of MCMC chain to be disregarded
    sampling_rate : ...
    a, b : prior parameters of d, the intrindic dimensionality distribution
    c : prior parameters of p
    f : parameters of zeta
    """

    def __init__(
        self,
        metric="euclidean",
        K=2,
        zeta=0.8,
        q=3,
        n_iter=10000,
        n_random_starts=1,
        burn_in=0.9,
        sampling_rate=10,
        a=None,
        b=None,
        c=None,
        f=None,
    ):
        self.metric = metric
        self.K = K
        self.zeta = zeta
        self.q = q
        self.n_iter = n_iter
        self.n_random_starts = n_random_starts
        self.burn_in = burn_in
        self.sampling_rate = sampling_rate
        self.a = np.ones(K) if a is None else a
        self.b = np.ones(K) if b is None else b
        self.c = np.ones(K) if c is None else c
        self.f = np.ones(K) if f is None else f
        self.n_samp = np.floor(
            (self.n_iter - np.ceil(self.burn_in * self.n_iter)) / self.sampling_rate
        ).astype(int)

        super(Hidalgo, self).__init__()

    def _find_neighbors(self, X):
        """."""
        self.N_, self.d_ = np.shape(X)
        self._neighbors = NearestNeighbors(
            n_neighbors=3, algorithm="ball_tree", metric=self.metric
        )
        self._fitted_neighbors = self._neighbors.fit(X)
        self.distances_, _ = self._fitted_neighbors.kneighbors(X)
        self.mu_ = np.sort(np.divide(self.distances_[:, 2], self.distances_[:, 1]))

        return self

    def _gibbs_sampling(self, X):
        """."""
        self.n_par = self.N_ + 2 * self.K + 2  # ??

        ######################################
        # Gibbs sampling algorithm with known likelihood function and
        # specified prior distributions
        # output is sampling array of size i.e. self.sampling = self.n_samp x self.n_par
        # self.sampling represents discrete samples of the
        # (unknown) target distribution, columns corresponding to parameters
        ######################################

        # random starts (should) run in parallel

        # need to decide if running/compiling C code for this
        # or implementing Gibbs in python where we either:
        # - interface pymc, however there is a custom (likelihood) function added in this algo
        # - write entire sampling algo again

        return self

    def _fit(self, X):
        """."""
        self._find_neighbors(X)
        self._gibbs_sampling(X)

        # sampling comes from the executable file in this order
        self.d_ = np.mean(self.sampling[:, : self.K], axis=0)
        self.derr_ = np.std(self.sampling[:, : self.K], axis=0)
        self.p_ = np.mean(self.sampling[:, self.K : 2 * self.K], axis=0)
        self.perr_ = np.std(self.sampling[:, self.K : 2 * self.K], axis=0)
        self.lik_ = np.mean(self.sampling[:, -1], axis=0)
        self.likerr_ = np.std(self.sampling[:, -1], axis=0)

        # other probability, pi computed here

        # main output of interest here is self.d_ = N x 1 array
        # representing the segmentation corresponding to each row of data

        return self
