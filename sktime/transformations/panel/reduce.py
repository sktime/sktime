#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["Tabularizer"]

import pandas as pd
from sklearn.utils.validation import check_array

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.data_processing import from_2d_array_to_nested
from sktime.utils.data_processing import from_3d_numpy_to_2d_array
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.utils.validation.panel import check_X


class Tabularizer(_PanelToTabularTransformer):
    """
    A transformer that turns time series/panel data into tabular data.

    This estimator converts nested pandas dataframe containing
    time-series/panel data with numpy arrays or pandas Series in
    dataframe cells into a tabular pandas dataframe with only primitives in
    cells. This is useful for transforming
    time-series/panel data into a format that is accepted by standard
    validation learning algorithms (as in sklearn).
    """

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
        if isinstance(X, pd.DataFrame):
            return from_nested_to_2d_array(X)
        else:
            return from_3d_numpy_to_2d_array(X)

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
        # We expect a tabular pd.DataFrame or np.array here, hence we use
        # scikit-learn's input validation function.
        X = check_array(X)
        return from_2d_array_to_nested(X)
