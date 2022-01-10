# -*- coding: utf-8 -*-
from typing import Callable, Union

import numpy as np
from numpy.random import RandomState

from sktime.clustering_redo.partitioning._lloyds import _Lloyds
from sktime.distances import pairwise_distance


class KMedoids(_Lloyds):
    """Time series K-medoids implementation using Lloyds algorithm."""

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, Callable] = "forgy",
        metric: Union[str, Callable] = "dtw",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
    ):
        self._precomputed_pairwise = None

        super(KMedoids, self).__init__(
            n_clusters,
            init_algorithm,
            metric,
            n_init,
            max_iter,
            tol,
            verbose,
            random_state,
        )

    def _fit(self, X: np.ndarray, y=None) -> np.ndarray:
        self._precomputed_pairwise = pairwise_distance(X, metric=self.metric)
        return super()._fit(X, y)

    def _compute_new_cluster_centers(
        self, X: np.ndarray, assignment_indexes: np.ndarray
    ) -> np.ndarray:
        """Compute new centers.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Time series instances to predict their cluster indexes.
        assignment_indexes: np.ndarray
            Indexes that each time series in X belongs to.

        Returnist(zip(curr_indexes, curr_indexes))s
        -------
        np.ndarray (3d of shape (n_clusters, n_dimensions, series_length)
            New cluster center values.
        """
        new_centers = np.zeros((self.n_clusters, X.shape[1], X.shape[2]))
        for i in range(self.n_clusters):
            curr_indexes = np.where(assignment_indexes == i)[0]
            distance_matrix = np.zeros((len(curr_indexes), len(curr_indexes)))
            for j in range(len(curr_indexes)):
                for k in range(len(curr_indexes)):
                    distance_matrix[j, k] = self._precomputed_pairwise[j, k]
            new_centers[i, :] = X[curr_indexes[np.argmin(sum(distance_matrix))]]
        return new_centers
