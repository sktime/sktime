# -*- coding: utf-8 -*-
from typing import Callable, Union

import numpy as np
from numpy.random import RandomState

from sktime.clustering_redo.metrics._averaging import resolve_average_callable
from sktime.clustering_redo.partitioning._lloyds import _Lloyds


class KMeans(_Lloyds):
    """Time series K mean implementation."""

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
        averaging_technique: Union[str, Callable] = "mean",
    ):
        self.averaging_technique = averaging_technique
        self._average_technique = resolve_average_callable(averaging_technique)

        super(KMeans, self).__init__(
            n_clusters,
            init_algorithm,
            metric,
            n_init,
            max_iter,
            tol,
            verbose,
            random_state,
        )

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

        Returns
        -------
        np.ndarray (3d of shape (n_clusters, n_dimensions, series_length)
            New cluster center values.
        """
        new_centers = np.zeros((self.n_clusters, X.shape[1], X.shape[2]))
        for i in range(self.n_clusters):
            curr_indexes = np.where((assignment_indexes == i))
            new_centers[i, :] = self._average_technique(X[curr_indexes])
        return new_centers
