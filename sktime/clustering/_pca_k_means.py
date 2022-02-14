# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState

from sktime.clustering._base import TimeSeriesInstances
from sktime.clustering._k_means import TimeSeriesKMeans
from sktime.clustering.metrics.averaging._averaging import resolve_average_callable
from sktime.transformations.panel.pca import PCATransformer


class TimeSeriesPcaKMeans(TimeSeriesKMeans):
    """Time series PCA K-mean implementation.

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

    averaging_method: str or Callable, defaults = 'mean'
        Averaging method to compute the average of a cluster. Any of the following
        strings are valid: ['mean']. If a Callable is provided must take the form
        Callable[[np.ndarray], np.ndarray].
    distance_params: dict, defaults = None
        Dictionary containing kwargs for the distance metric being used.
    pca_params :  dict, defaults = None
        Dictionary containing kwargs for pca. See sklearn pca documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        for valid arguments.

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
        "capability:multivariate": False,
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
        averaging_method: Union[str, Callable[[np.ndarray], np.ndarray]] = "mean",
        distance_params: dict = None,
        pca_params: dict = None,
    ):
        self.averaging_method = averaging_method
        self.pca_params = pca_params
        self._averaging_method = resolve_average_callable(averaging_method)

        if pca_params is None:
            pca_params = {}
        self._pca = PCATransformer(random_state=random_state, **pca_params)
        super(TimeSeriesKMeans, self).__init__(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            distance_params=distance_params,
        )

    def _fit(self, X: TimeSeriesInstances, y=None):
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
        _X = self._check_clusterer_input(self._pca.fit_transform(X))
        return super(TimeSeriesPcaKMeans, self)._fit(_X)

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
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
        _X = self._check_clusterer_input(self._pca.fit_transform(X))
        return super(TimeSeriesPcaKMeans, self)._predict(_X)
