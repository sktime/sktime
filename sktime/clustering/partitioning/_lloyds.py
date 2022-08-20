# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum

from sktime.clustering.base import BaseClusterer
from sktime.clustering.metrics.averaging import mean_average
from sktime.distances import distance_factory, pairwise_distance
from sktime.distances._ddtw import average_of_slope_transform


def _forgy_center_initializer(
    X: np.ndarray, n_clusters: int, random_state: np.random.RandomState, **kwargs
) -> np.ndarray:
    """Compute the initial centers using forgy method.

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_instances,n_dimensions,series_length))
        Time series instances to cluster.
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    random_state: np.random.RandomState
        Determines random number generation for centroid initialization.

    Returns
    -------
    np.ndarray (3d array of shape (n_clusters, n_dimensions, series_length))
        Indexes of the cluster centers.
    """
    return X[random_state.choice(X.shape[0], n_clusters, replace=False)]


def _random_center_initializer(
    X: np.ndarray, n_clusters: int, random_state: np.random.RandomState, **kwargs
) -> np.ndarray:
    """Compute initial centroids using random method.

    This works by assigning each point randomly to a cluster. Then the average of
    the cluster is taken to get the centers.

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_instances,n_dimensions,series_length))
        Time series instances to cluster.
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    random_state: np.random.RandomState
        Determines random number generation for centroid initialization.

    Returns
    -------
    np.ndarray (3d array of shape (n_clusters, n_dimensions, series_length))
        Indexes of the cluster centers.
    """
    new_centres = np.zeros((n_clusters, X.shape[1], X.shape[2]))
    selected = random_state.choice(n_clusters, X.shape[0], replace=True)
    for i in range(n_clusters):
        curr_indexes = np.where(selected == i)[0]
        result = mean_average(X[curr_indexes])
        if result.shape[0] > 0:
            new_centres[i, :] = result

    return new_centres


def _kmeans_plus_plus(
    X: np.ndarray,
    n_clusters: int,
    random_state: np.random.RandomState,
    distance_metric: str = "euclidean",
    n_local_trials: int = None,
    distance_params: dict = None,
    **kwargs,
):
    """Compute initial centroids using kmeans++ method.

    This works by choosing one point at random. Next compute the distance between the
    center and each point. Sample these with a probability proportional to the square
    of the distance of the points from its nearest center.

    NOTE: This is adapted from sklearns implementation:
    https://
    github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/cluster/_kmeans.py

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_instances,n_dimensions,series_length))
        Time series instances to cluster.
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    random_state: np.random.RandomState
        Determines random number generation for centroid initialization.
    distance_metric: str, defaults = 'euclidean'
        String that is the distance metric.
    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    distance_params: dict, defaults = None
        Dictionary containing distance parameter kwargs.

    Returns
    -------
    np.ndarray (3d array of shape (n_clusters, n_dimensions, series_length))
        Indexes of the cluster centers.
    """
    n_samples, n_timestamps, n_features = X.shape

    centers = np.empty((n_clusters, n_timestamps, n_features), dtype=X.dtype)
    n_samples, n_timestamps, n_features = X.shape

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    if distance_params is None:
        distance_params = {}

    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]
    closest_dist_sq = (
        pairwise_distance(centers[0, np.newaxis], X, metric=distance_metric) ** 2
    )
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = (
            pairwise_distance(X[candidate_ids], X, metric=distance_metric) ** 2
        )

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]

    return centers


class TimeSeriesLloyds(BaseClusterer, ABC):
    """Abstact class that implements time series Lloyds algorithm.

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

    _tags = {
        "capability:multivariate": True,
    }

    _init_algorithms = {
        "forgy": _forgy_center_initializer,
        "random": _random_center_initializer,
        "kmeans++": _kmeans_plus_plus,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, Callable] = "random",
        metric: Union[str, Callable] = "euclidean",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
        distance_params: dict = None,
    ):
        self.init_algorithm = init_algorithm
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._init_algorithm = None

        self._distance_params = distance_params
        if distance_params is None:
            self._distance_params = {}

        super(TimeSeriesLloyds, self).__init__(n_clusters=n_clusters)

    def _check_params(self, X: np.ndarray) -> None:
        """Check parameters are valid and initialized.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Time series instances to cluster.

        Raises
        ------
        ValueError
            If the init_algorithm value is invalid.
        """
        self._random_state = check_random_state(self.random_state)

        if isinstance(self.init_algorithm, str):
            self._init_algorithm = self._init_algorithms.get(self.init_algorithm)
        else:
            self._init_algorithm = self.init_algorithm

        if not isinstance(self._init_algorithm, Callable):
            raise ValueError(
                f"The value provided for init_algorim: {self.init_algorithm} is "
                f"invalid. The following are a list of valid init algorithms strings: "
                f"{list(self._init_algorithms.keys())}"
            )

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params

        self._distance_metric = distance_factory(
            X[0], X[1], metric=self.metric, **self._distance_params
        )

    def _fit(self, X: np.ndarray, y=None):
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
        if self.metric == "ddtw" or self.metric == "wddtw":
            X = average_of_slope_transform(X)
            if self.metric == "ddtw":
                self._distance_metric = distance_factory(
                    X[0], X[1], metric="dtw", **self._distance_params
                )
            else:
                self._distance_metric = distance_factory(
                    X[0], X[1], metric="wdtw", **self._distance_params
                )
        else:
            self._distance_metric = distance_factory(
                X[0], X[1], metric=self.metric, **self._distance_params
            )
        best_centers = None
        best_inertia = np.inf
        best_labels = None
        best_iters = self.max_iter
        for _ in range(self.n_init):
            labels, centers, inertia, n_iters = self._fit_one_init(X)
            if inertia < best_inertia:
                best_centers = centers
                best_labels = labels
                best_inertia = inertia
                best_iters = n_iters

        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.n_iter_ = best_iters
        return self

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Time series instances to predict their cluster indexes.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        if self.metric == "ddtw" or self.metric == "wddtw":
            X = average_of_slope_transform(X)
        return self._assign_clusters(X, self.cluster_centers_)[0]

    def _fit_one_init(self, X) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Perform one pass of kmeans.

        This is done because the initial center assignment greatly effects the final
        result so we perform multiple passes at kmeans with different initial center
        assignments and keep the best results going froward.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Training time series instances to cluster.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instance,))
            Labels that is the index each time series belongs to.
        np.ndarray (3d array of shape (n_clusters, n_dimensions,
            series_length))
            Time series that represent each of the cluster centres. If the algorithm
            stops before fully converging these will not be consistent with labels.
        float
            Sum of squared distances of samples to their closest cluster center,
            weighted by the sample weights if provided.
        """
        cluster_centres = self._init_algorithm(
            X,
            self.n_clusters,
            self._random_state,
            distance_metric=self._distance_metric,
        )
        old_inertia = np.inf
        old_labels = None
        for i in range(self.max_iter):
            labels, inertia = self._assign_clusters(
                X,
                cluster_centres,
            )

            if np.abs(old_inertia - inertia) < self.tol:
                break
            old_inertia = inertia

            if np.array_equal(labels, old_labels):
                if self.verbose:
                    print(  # noqa: T001
                        f"Converged at iteration {i}: strict convergence."
                    )
                break
            old_labels = labels

            cluster_centres = self._compute_new_cluster_centers(X, labels)

            if self.verbose is True:
                print(f"Iteration {i}, inertia {inertia}.")  # noqa: T001

        labels, inertia = self._assign_clusters(X, cluster_centres)
        centres = cluster_centres

        return labels, centres, inertia, i + 1

    def _assign_clusters(
        self, X: np.ndarray, cluster_centres: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Assign each instance to a cluster.

        This is done by computing the distance between each instance and
        each cluster. For each instance an index is returned that indicates
        which center had the smallest distance to it.

        Parameters
        ----------
        X : np.ndarray (3d array of shape (n_instances, n_dimensions, series_length))
            Time series instances to predict their cluster indexes.
        cluster_centres: np.ndarray (3d array of shape
                                        (n_clusters, n_dimensions, series_length))
            Cluster centers to assign to.

        Returns
        -------
        np.ndarray (1d array of shape (n_instance,))
            Array of indexes of each instance closest cluster.
        float
            Only returned when return_inertia is true. Float representing inertia of
            the assigned clusters.
        """
        pairwise = pairwise_distance(
            X, cluster_centres, metric=self.metric, **self._distance_params
        )
        return pairwise.argmin(axis=1), pairwise.min(axis=1).sum()

    def _score(self, X, y=None):
        return -self.inertia_

    @abstractmethod
    def _compute_new_cluster_centers(
        self, X: np.ndarray, assignment_indexes: np.ndarray
    ) -> np.ndarray:
        """Abstract method to compute new centers.

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
        ...
