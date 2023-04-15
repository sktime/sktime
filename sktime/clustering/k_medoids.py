# -*- coding: utf-8 -*-
"""Time series kmedoids."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState

from sktime.clustering.metrics.medoids import medoids
from sktime.clustering.partitioning import TimeSeriesLloyds
from sktime.distances import pairwise_distance


class TimeSeriesKMedoids(TimeSeriesLloyds):
    """Time series K-medoids implementation.

    Parameters
    ----------
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init_algorithm: str, defaults = 'forgy'
        Method for initializing cluster centers. Any of the following are valid:
        ['kmeans++', 'random', 'forgy']
    metric: str or Callable, defaults = 'dtw'
        Distance metric to compute similarity between time series. Any of the following
        are valid: ['dtw', 'euclidean', 'erp', 'edr', 'lcss', 'squared', 'ddtw', 'wdtw',
        'wddtw']
    n_init: int, defaults = 10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter: int, defaults = 30
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, defaults = 1e-6
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose: bool, defaults = False
        Verbosity mode.
    random_state: int or np.random.RandomState instance or None, defaults = None
        Determines random number generation for centroid initialization.
    distance_params: dict, defaults = None
        Dictonary containing kwargs for the distance metric being used.

    Attributes
    ----------
    cluster_centers_: np.ndarray (3d array of shape (n_clusters, n_dimensions,
        series_length))
        Time series that represent each of the cluster centers. If the algorithm stops
        before fully converging these will not be consistent with labels_.
    labels_: np.ndarray (1d array of shape (n_instance,))
        Labels that is the index each time series belongs to.
    inertia_: float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_: int
        Number of iterations run.
    """

    _tags = {"python_dependencies": "numba"}

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, Callable] = "random",
        metric: Union[str, Callable] = "dtw",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
        distance_params: dict = None,
    ):
        self._precomputed_pairwise = None

        super(TimeSeriesKMedoids, self).__init__(
            n_clusters,
            init_algorithm,
            metric,
            n_init,
            max_iter,
            tol,
            verbose,
            random_state,
            distance_params,
        )

    def _fit(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Training time series instances to cluster.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        self._check_params(X)
        self._precomputed_pairwise = pairwise_distance(
            X, metric=self.metric, **self._distance_params
        )
        return super()._fit(X, y)

    def _compute_new_cluster_centers(
        self, X: np.ndarray, assignment_indexes: np.ndarray
    ) -> np.ndarray:
        """Compute new centers.

        Parameters
        ----------
        X : np.ndarray (3d array of shape (n_instances, n_dimensions, series_length))
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
            curr_indexes = np.where(assignment_indexes == i)[0]
            distance_matrix = np.zeros((len(curr_indexes), len(curr_indexes)))
            for j in range(len(curr_indexes)):
                curr_j = curr_indexes[j]
                for k in range(len(curr_indexes)):
                    distance_matrix[j, k] = self._precomputed_pairwise[
                        curr_j, curr_indexes[k]
                    ]
            result = medoids(
                X[curr_indexes], precomputed_pairwise_distance=distance_matrix
            )
            if result.shape[0] > 0:
                new_centers[i, :] = result
        return new_centers

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {
            "n_clusters": 2,
            "init_algorithm": "random",
            "metric": "euclidean",
            "n_init": 1,
            "max_iter": 1,
            "tol": 0.0001,
            "verbose": False,
            "random_state": 1,
        }
