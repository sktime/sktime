"""KNeighbors Time Series Estimator base class."""

__author__ = ["fkiraly", "Z-Fran"]
__all__ = ["_BaseKnnTimeSeriesEstimator"]

import numpy as np

# add new distance string codes here
DISTANCES_SUPPORTED = [
    "euclidean",
    # Euclidean will default to the base class distance
    "squared",
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "lcss",
    "edr",
    "erp",
    "msm",
    "twe",
]


class _BaseKnnTimeSeriesEstimator:
    """Base KNeighbors Time Series Estimator."""

    def __init__(
        self,
        n_neighbors=1,
        weights="uniform",
        algorithm="brute",
        distance="dtw",
        distance_params=None,
        distance_mtype=None,
        pass_train_distances=False,
        leaf_size=30,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.distance = distance
        self.distance_params = distance_params
        self.distance_mtype = distance_mtype
        self.pass_train_distances = pass_train_distances
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

        super().__init__()

        # input check for supported distance strings
        if isinstance(distance, str) and distance not in DISTANCES_SUPPORTED:
            raise ValueError(
                f"Unrecognised distance measure string: {distance}. "
                f"Allowed values for string codes are: {DISTANCES_SUPPORTED}. "
                "Alternatively, pass a callable distance measure into the constructor."
            )

        knn_cls = self._knn_cls

        if algorithm == "brute_incr":
            # brute_incr is not a valid sklearn algorithm
            _algorithm = "brute"
        else:
            _algorithm = algorithm

        self.knn_estimator_ = knn_cls(
            n_neighbors=n_neighbors,
            algorithm=_algorithm,
            metric="precomputed",
            leaf_size=leaf_size,
            n_jobs=n_jobs,
            weights=weights,
        )

        # the distances in sktime.distances want numpy3D
        #   otherwise all Panel formats are ok
        if isinstance(distance, str):
            self.set_tags(X_inner_mtype="numpy3D")
            self.set_tags(**{"capability:unequal_length": False})
            self.set_tags(**{"capability:missing_values": False})
        elif distance_mtype is not None:
            self.set_tags(X_inner_mtype=distance_mtype)

        from sktime.dists_kernels import BasePairwiseTransformerPanel

        # inherit capability tags from distance, if it is an estimator
        if isinstance(distance, BasePairwiseTransformerPanel):
            inherit_tags = [
                "capability:missing_values",
                "capability:unequal_length",
                "capability:multivariate",
            ]
            self.clone_tags(distance, inherit_tags)

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : sktime compatible Panel data container, of mtype X_inner_mtype,
            with n time series to fit the estimator to
        y : {array-like, sparse matrix}
            Target values of shape = [n]
        """
        # internal import to avoid circular imports
        from sktime.dists_kernels.base.adapters._sklearn import _SklearnDistanceAdapter

        self._dist_adapt = _SklearnDistanceAdapter(
            distance=self.distance,
            distance_params=self.distance_params,
            n_vars=X.shape[1],
            is_equal_length=self._X_metadata["is_equal_length"],
        )
        if self.algorithm == "brute":
            return self._fit_precomp(X=X, y=y)
        else:
            return self._fit_dist(X=X, y=y)

    def _fit_dist(self, X, y):
        """Fit the model using adapted distance metric."""
        # use distance adapter, see _BaseKnnTimeSeriesEstimator, _SklearnDistanceAdapter
        metric = self._dist_adapt

        self.knn_estimator_.set_params(metric=metric)

        X = self._dist_adapt._convert_X_to_sklearn(X)
        self.knn_estimator_.fit(X, y)
        return self

    def _fit_precomp(self, X, y):
        """Fit the model using precomputed distance matrix."""
        # store full data as indexed X
        self._X = X

        if self.pass_train_distances:
            dist_mat = self._dist_adapt._distance(X)
        else:
            n = self._X_metadata["n_instances"]
            # if we do not want/need to pass train-train distances,
            #   we still need to pass a zeros matrix, this means "do not consider"
            # citing the sklearn KNeighborsClassifier docs on distance matrix input:
            # "X may be a sparse graph, in which case only "nonzero" elements
            #   may be considered neighbors."
            dist_mat = np.zeros([n, n], dtype="float")

        self.knn_estimator_.fit(dist_mat, y)

        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : sktime-compatible data format, Panel or Series, with n_samples series
        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        # self._X should be the stored _X
        dist_mat = self._dist_adapt._distance(X, self._X)

        result = self.knn_estimator_.kneighbors(
            dist_mat, n_neighbors=n_neighbors, return_distance=return_distance
        )

        # result is either dist, or (dist, ind) pair, depending on return_distance
        return result

    def _predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : sktime-compatible Panel data, of mtype X_inner_mtype, with n_samples series
            data to predict class labels for

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        if self.algorithm == "brute":
            return self._predict_precomp(X)
        else:
            return self._predict_dist(X)

    def _predict_dist(self, X):
        """Predict using adapted distance metric."""
        X = self._dist_adapt._convert_X_to_sklearn(X)
        y_pred = self.knn_estimator_.predict(X)
        return y_pred

    def _predict_precomp(self, X):
        """Predict using precomputed distance matrix."""
        # self._X should be the stored _X
        dist_mat = self._dist_adapt._distance(X, self._X)
        y_pred = self.knn_estimator_.predict(dist_mat)
        return y_pred
