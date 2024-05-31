"""Implements adapter for sklearn models."""

__author__ = ["Z-Fran"]
__all__ = ["KNeighborsSklearnAdapter"]

import numpy as np
import pandas as pd

from sktime.datatypes import convert


class KNeighborsSklearnAdapter:
    """Mixin adapter class for sklearn KNeighbors models."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["Z-Fran"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
    }

    def _one_element_distance_npdist(self, x, y, n_vars=None):
        if n_vars is None:
            n_vars = self.n_vars_
        x = np.reshape(x, (1, n_vars, -1))
        y = np.reshape(y, (1, n_vars, -1))
        return self._distance(x, y)[0, 0]

    def _one_element_distance_sktime_dist(self, x, y, n_vars=None):
        if n_vars is None:
            n_vars = self.n_vars_
        if n_vars == 1:
            x = np.reshape(x, (1, n_vars, -1))
            y = np.reshape(y, (1, n_vars, -1))
        elif self._X_metadata["is_equal_length"]:
            x = np.reshape(x, (-1, n_vars))
            y = np.reshape(y, (-1, n_vars))
            x_ix = pd.MultiIndex.from_product([[0], range(len(x))])
            y_ix = pd.MultiIndex.from_product([[0], range(len(y))])
            x = pd.DataFrame(x, index=x_ix)
            y = pd.DataFrame(y, index=y_ix)
        else:  # multivariate, unequal length
            # in _convert_X_to_sklearn, we have encoded the length as the first column
            # this was coerced to float, so we round to avoid rounding errors
            x_len = round(x[0])
            y_len = round(y[0])
            # pd.pivot switches the axes, compared to numpy
            x = np.reshape(x[1:], (n_vars, -1)).T
            y = np.reshape(y[1:], (n_vars, -1)).T
            # cut to length
            x = x[:x_len]
            y = y[:y_len]
            x_ix = pd.MultiIndex.from_product([[0], range(x_len)])
            y_ix = pd.MultiIndex.from_product([[0], range(y_len)])
            x = pd.DataFrame(x, index=x_ix)
            y = pd.DataFrame(y, index=y_ix)
        return self._distance(x, y)[0, 0]

    def _convert_X_to_sklearn(self, X):
        """Convert X to 2D numpy for sklearn."""
        # special treatment for unequal length series
        if not self._X_metadata["is_equal_length"]:
            # then we know we are dealing with pd-multiindex
            # as a trick to deal with unequal length data,
            # we flatten encode the length as the first column
            X_w_ix = X.reset_index(-1)
            X_pivot = X_w_ix.pivot(columns=[X_w_ix.columns[0]])
            # fillna since this creates nan but sklearn does not accept these
            # the fill value does not matter as the distance ignores it
            X_pivot = X_pivot.fillna(0).to_numpy()
            X_lens = X.groupby(X_w_ix.index).size().to_numpy()
            # add the first column, encoding length of individual series
            X_w_lens = np.concatenate([X_lens[:, None], X_pivot], axis=1)
            return X_w_lens

        # equal length series case
        if isinstance(X, np.ndarray):
            X_mtype = "numpy3D"
        else:
            X_mtype = "pd-multiindex"
        return convert(X, from_type=X_mtype, to_type="numpyflat")

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : sktime compatible Panel data container, of mtype X_inner_mtype,
            with n time series to fit the estimator to
        y : {array-like, sparse matrix}
            Target values of shape = [n]
        """
        self.n_vars_ = X.shape[1]
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
        dist_mat = self._distance(X, self._X)

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
        X = self._convert_X_to_sklearn(X)
        y_pred = self.knn_estimator_.predict(X)
        return y_pred

    def _predict_precomp(self, X):
        """Predict using precomputed distance matrix."""
        # self._X should be the stored _X
        dist_mat = self._distance(X, self._X)
        y_pred = self.knn_estimator_.predict(dist_mat)
        return y_pred
