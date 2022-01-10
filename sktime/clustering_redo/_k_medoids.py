# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering_redo.partitioning._lloyds import _Lloyds
from sktime.distances import pairwise_distance


class KMedoids(_Lloyds):
    """Time series K-medoids implementation using Lloyds algorithm."""

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
            distance_matrix = pairwise_distance(X[curr_indexes], metric=self.metric)
            new_centers[i, :] = np.argmin(sum(distance_matrix))
        return new_centers
