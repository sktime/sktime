"""KNeighbors Time Series Estimator base class."""

__author__ = ["fkiraly", "Z-Fran"]
__all__ = ["_BaseKnnTimeSeriesEstimator"]


class _BaseKnnTimeSeriesEstimator:
    """Base KNeighbors Time Series Estimator."""

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
