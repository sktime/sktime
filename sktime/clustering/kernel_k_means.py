# -*- coding: utf-8 -*-
"""Time series kernel kmeans."""
from typing import Dict, Union

import numpy as np
from numpy.random import RandomState

from sktime.clustering.base import BaseClusterer, TimeSeriesInstances
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("tslearn", severity="warning")


class TimeSeriesKernelKMeans(BaseClusterer):
    """Kernel algorithm wrapper tslearns implementation.

    Parameters
    ----------
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    kernel : string, or callable (default: "gak")
        The kernel should either be "gak", in which case the Global Alignment
        Kernel from [2]_ is used or a value that is accepted as a metric
        by `scikit-learn's pairwise_kernels
        <https://scikit-learn.org/stable/modules/generated/\
        sklearn.metrics.pairwise.pairwise_kernels.html>`_
    n_init: int, defaults = 10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    kernel_params : dict or None (default: None)
        Kernel parameters to be passed to the kernel function.
        None means no kernel parameter is set.
        For Global Alignment Kernel, the only parameter of interest is `sigma`.
        If set to 'auto', it is computed based on a sampling of the training
        set
        (cf :ref:`tslearn.metrics.sigma_gak <fun-tslearn.metrics.sigma_gak>`).
        If no specific value is set for `sigma`, its defaults to 1.
    max_iter: int, defaults = 300
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, defaults = 1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose: bool, defaults = False
        Verbosity mode.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for GAK cross-similarity matrix
        computations.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.
    random_state: int or np.random.RandomState instance or None, defaults = None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
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
        "python_dependencies": "tslearn",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        kernel: str = "gak",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        kernel_params: Union[dict, None] = None,
        verbose: bool = False,
        n_jobs: Union[int, None] = None,
        random_state: Union[int, RandomState] = None,
    ):
        self.kernel = kernel
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._tslearn_kernel_k_means = None

        super(TimeSeriesKernelKMeans, self).__init__(n_clusters=n_clusters)

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
        from tslearn.clustering import KernelKMeans as TsLearnKernelKMeans

        verbose = 0
        if self.verbose is True:
            verbose = 1

        if self._tslearn_kernel_k_means is None:
            self._tslearn_kernel_k_means = TsLearnKernelKMeans(
                n_clusters=self.n_clusters,
                kernel=self.kernel,
                max_iter=self.max_iter,
                tol=self.tol,
                n_init=self.n_init,
                kernel_params=self.kernel_params,
                n_jobs=self.n_jobs,
                verbose=verbose,
                random_state=self.random_state,
            )
        self._tslearn_kernel_k_means.fit(X)
        self.labels_ = self._tslearn_kernel_k_means.labels_
        self.inertia_ = self._tslearn_kernel_k_means.inertia_
        self.n_iter_ = self._tslearn_kernel_k_means.n_iter_

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
        return self._tslearn_kernel_k_means.predict(X)

    @classmethod
    def get_test_params(cls, parameter_set="default") -> Dict:
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
            "kernel": "gak",
            "n_init": 1,
            "max_iter": 1,
            "tol": 0.0001,
            "kernel_params": None,
            "verbose": False,
            "n_jobs": 1,
            "random_state": 1,
        }

    def _score(self, X, y=None) -> float:
        return np.abs(self.inertia_)
