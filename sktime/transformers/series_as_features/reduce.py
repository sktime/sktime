#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["Tabularizer"]

from sklearn.utils.validation import check_array

from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import from_2d_numpy_to_nested
from sktime.utils.data_container import from_nested_to_2d_numpy
from sktime.utils.data_container import get_time_index
from sktime.utils.validation.series_as_features import check_X


class Tabularizer(BaseSeriesAsFeaturesTransformer):
    """
    A transformer that turns time series/panel data into tabular data.

    This estimator converts nested pandas dataframe containing
    time-series/panel data with numpy arrays or pandas Series in
    dataframe cells into a tabular pandas dataframe with only primitives in
    cells. This is useful for transforming
    time-series/panel data into a format that is accepted by standard
    validation learning algorithms (as in sklearn).
    """

    def fit(self, X, y=None):
        self._columns = X.columns
        self._index = X.index
        self._time_index = get_time_index(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Transform nested pandas dataframe into tabular dataframe.

        Parameters
        ----------
        X : pandas DataFrame
            Nested dataframe with pandas series or numpy arrays in cells.
        y : array-like, optional (default=None)

        Returns
        -------
        Xt : pandas DataFrame
            Transformed dataframe with only primitives in cells.
        """
        self.check_is_fitted()
        X = check_X(X)
        return from_nested_to_2d_numpy(X)

    def inverse_transform(self, X, y=None):
        """Transform tabular pandas dataframe into nested dataframe.

        Parameters
        ----------
        X : pandas DataFrame
            Tabular dataframe with primitives in cells.
        y : array-like, optional (default=None)

        Returns
        -------
        Xt : pandas DataFrame
            Transformed dataframe with series in cells.
        """
        self.check_is_fitted()
        if len(self._columns) > 1:
            raise NotImplementedError(
                f"`inverse-transform` currently only "
                f"handles univariate data, but found: "
                f"{len(self._columns)} columns in `fit`"
            )

            # we expect a tabular pd.DataFrame or np.array here, hence we use
            # scikit-learn's input validation function
        X = check_array(
            X,
        )

        Xt = from_2d_numpy_to_nested(X, index=self._index, time_index=self._time_index)
        Xt.columns = self._columns
        return Xt
