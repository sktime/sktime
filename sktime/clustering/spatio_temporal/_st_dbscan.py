"""Spatiotemporal DBSCAN."""

__author__ = ["eren-ck", "vagechirkov"]

import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sktime.clustering import BaseClusterer
from sktime.datatypes import update_data
from sktime.utils.warnings import warn


class STDBSCAN(BaseClusterer):
    """
    Spatiotemporal DBSCAN clustering.

    Clusters data based on specified spatial and temporal proximity thresholds.

    Parameters
    ----------
    eps1 : float, default=0.5
        Maximum spatial distance for points to be considered related.
    eps2 : float, default=10
        Maximum temporal distance for points to be considered related.
    min_samples : int, default=5
        Minimum number of samples to form a core point.
    metric : str, default='euclidean'
        Distance metric to use; options include 'euclidean', 'manhattan',
        'chebyshev', etc.
    sparse_matrix_threshold : int, default=20_000
        Sets the limit on the number of samples for which the algorithm can
        efficiently compute distances with a full matrix approach. Datasets
        exceeding this threshold will be handled using sparse matrix methods.
    n_jobs : int or None, default=-1
        Number of parallel jobs for distance computation; -1 uses all cores.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point; noise is labeled as -1.

    References
    ----------
    .. [1] Birant, D., & Kut, A. "ST-DBSCAN: An algorithm for clustering
       spatial-temporal data." Data Knowl. Eng., vol. 60, no. 1, pp. 208-221, Jan. 2007,
       doi: [10.1016/j.datak.2006.01.013](https://doi.org/10.1016/j.datak.2006.01.013).
    .. [2] Cakmak, E., Plank, M., Calovi, D. S., Jordan, A., & Keim, D. "Spatio-temporal
       clustering benchmark for collective animal behavior." ACM, Nov. 2021, pp. 5-8.
       doi: [10.1145/3486637.3489487](https://doi.org/10.1145/3486637.3489487).
    """

    _tags = {
        "maintainers": "vagechirkov",
        "authors": ["eren-ck", "vagechirkov"],
        "python_dependencies": ["scipy", "scikit-learn"],
        "X_inner_mtype": "numpyflat",
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:multithreading": True,
        "capability:predict": True,
        "capability:predict_proba": False,
        "capability:out_of_sample": True,
    }

    DELEGATED_FITTED_PARAMS = ["core_sample_indices_", "components_ ", "labels_"]

    def __init__(
        self,
        eps1=0.5,
        eps2=10,
        min_samples=5,
        metric="euclidean",
        sparse_matrix_threshold=20_000,
        n_jobs=-1,
    ):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.metric = metric
        self.sparse_matrix_threshold = sparse_matrix_threshold
        self.n_jobs = n_jobs
        self.dbscan_ = None
        super().__init__()

    def _fit(self, X, y=None):
        """
        Apply the ST-DBSCAN algorithm to cluster spatiotemporal data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data with the first column representing time as a float.
            The remaining columns represent spatial coordinates. Example format
            for a 2D dataset:
                [[time_step1, x, y],
                 [time_step2, x, y],
                 ...]
        y : ignored, exists for API consistency reasons

        Returns
        -------
        self :
            Fitted instance with cluster labels.
        """
        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError("eps1, eps2, minPts must be positive")

        self._X = X

        n, m = X.shape

        if len(X) < self.sparse_matrix_threshold:
            # compute with quadratic memory consumption

            # Compute squared form Distance Matrix
            time_dist = pdist(X[:, 0].reshape(n, 1), metric=self.metric)
            spatial_dist = pdist(X[:, 1:], metric=self.metric)

            # filter the spatial_dist matrix using the time_dist
            dist = np.where(time_dist <= self.eps2, spatial_dist, 2 * self.eps1)

            self.dbscan_ = DBSCAN(
                eps=self.eps1, min_samples=self.min_samples, metric="precomputed"
            )
            self.dbscan_.fit(squareform(dist))

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # compute with sparse matrices
                # Compute sparse matrix for spatial distance
                nn_spatial = NearestNeighbors(
                    metric=self.metric, radius=self.eps1, n_jobs=self.n_jobs
                )
                nn_spatial.fit(X[:, 1:])
                euc_sp = nn_spatial.radius_neighbors_graph(X[:, 1:], mode="distance")

                # Compute sparse matrix for temporal distance
                nn_time = NearestNeighbors(
                    metric=self.metric, radius=self.eps2, n_jobs=self.n_jobs
                )
                nn_time.fit(X[:, 0].reshape(n, 1))
                time_sp = nn_time.radius_neighbors_graph(
                    X[:, 0].reshape(n, 1), mode="distance"
                )

                # combine both sparse matrices and filter by time distance matrix
                row = time_sp.nonzero()[0]
                column = time_sp.nonzero()[1]
                v = np.array(euc_sp[row, column])[0]

                # create sparse distance matrix
                dist_sp = coo_matrix((v, (row, column)), shape=(n, n))
                dist_sp = dist_sp.tocsc()
                dist_sp.eliminate_zeros()

                self.dbscan_ = DBSCAN(
                    eps=self.eps1, min_samples=self.min_samples, metric="precomputed"
                )
                self.dbscan_.fit(dist_sp)

        for key in self.DELEGATED_FITTED_PARAMS:
            if hasattr(self.dbscan_, key):
                setattr(self, key, getattr(self.dbscan_, key))

        return self

    def _predict(self, X, y=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : panel of time series, any sklearn Panel mtype
            Time series instances to predict cluster indexes for
        y: ignored, exists for API consistency reasons

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to
        """
        # if X is the same as seen in _fit, simply return the labels
        if X is self._X:
            return self.labels_
        else:
            all_X = update_data(X=self._X, X_new=X)
            warn(
                "sklearn and sktime DBSCAN estimators do not support different X "
                "in fit and predict, but a new X was passed in predict. "
                "Therefore, a clone of STDBSCAN will be fit, and results "
                "returned, without updating the state of the fitted estimator.",
                obj=self,
            )
            return self.clone().fit(all_X).labels_

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for clusterers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        param1 = {}

        param2 = {
            "eps1": 0.3,
            "eps2": 5,
            "min_samples": 3,
            "metric": "euclidean",
            "n_jobs": 1,
        }
        param3 = {
            "eps1": 0.5,
            "eps2": 10,
            "min_samples": 2,
            "metric": "manhattan",
            "n_jobs": 2,
        }

        return [param1, param2, param3]
