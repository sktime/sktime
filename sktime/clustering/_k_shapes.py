# -*- coding: utf-8 -*-
from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from sktime.clustering._base import BaseClusterer, TimeSeriesInstances
from sktime.clustering.partitioning._lloyds import (
    _forgy_center_initializer,
    _random_center_initializer,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("tslearn")
from tslearn.clustering import KShape  # noqa: E402


class TimeSeriesKShapes(BaseClusterer):
    """Kshape algorithm wrapper tslearns implementation.

    Parameters
    ----------
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init_algorithm: str, defaults = 'forgy'
        Method for initializing cluster centers. Any of the following are valid:
        ['kmeans++', 'random', 'forgy']
    n_init: int, defaults = 10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter: int, defaults = 30
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, defaults = 1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose: bool, defaults = False
        Verbosity mode.
    random_state: int or np.random.RandomState instance or None, defaults = None
        Determines random number generation for centroid initialization.

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
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, Callable] = "forgy",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
    ):
        self.n_clusters = n_clusters
        self.init_algorithm = init_algorithm
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._init_algorithm = None
        self._tslearn_k_shapes = None

        super(TimeSeriesKShapes, self).__init__()

    def _fit(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
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

        if self._tslearn_k_shapes is None:
            self._tslearn_k_shapes = KShape(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                n_init=self.n_init,
                verbose=self.verbose,
            )

        self._tslearn_k_shapes.fit(X)
        self._cluster_centers = self._tslearn_k_shapes.cluster_centers_
        self.labels_ = self._tslearn_k_shapes.labels_
        self.inertia_ = self._tslearn_k_shapes.inertia_
        self.n_iter_ = self._tslearn_k_shapes.n_iter_

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
        return self._tslearn_k_shapes.predict(X)

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

        if isinstance(self.init_algorithm, str) and self._init_algorithm is None:
            self._init_algorithm = self._init_algorithms.get(self.init_algorithm)
        else:
            self._init_algorithm = self.init_algorithm

        if not isinstance(self._init_algorithm, Callable):
            raise ValueError(
                f"The value provided for init_algorim: {self.init_algorithm} is "
                f"invalid. The following are a list of valid init algorithms strings: "
                f"{list(self._init_algorithms.keys())}"
            )

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {
            "n_clusters": 2,
            "init_algorithm": "random",
            "n_init": 1,
            "max_iter": 1,
            "tol": 1e-4,
            "verbose": False,
            "random_state": 1,
        }
        return params
