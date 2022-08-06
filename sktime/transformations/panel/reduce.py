# -*- coding: utf-8 -*-
"""Tabularizer transform, for pipelining."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "fkiraly"]
__all__ = ["Tabularizer"]

from sktime.datatypes import convert, convert_to
from sktime.transformations.base import BaseTransformer


class Tabularizer(BaseTransformer):
    """
    A transformer that turns time series/panel data into tabular data.

    This estimator converts nested pandas dataframe containing
    time-series/panel data with numpy arrays or pandas Series in
    dataframe cells into a tabular pandas dataframe with only primitives in
    cells. This is useful for transforming
    time-series/panel data into a format that is accepted by standard
    validation learning algorithms (as in sklearn).
    """

    _tags = {
        "fit_is_empty": True,
        "univariate-only": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["nested_univ", "numpy3D"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # and for y?
    }

    def _transform(self, X, y=None):
        """Transform nested pandas dataframe into tabular dataframe.

        Parameters
        ----------
        X : pandas DataFrame or 3D np.ndarray
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pandas DataFrame
            Transformed dataframe with only primitives in cells.
        """
        Xt = convert_to(X, to_type="numpyflat", as_scitype="Panel")
        return Xt

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
        Xt = convert(X, from_type="numpyflat", to_type="numpy3D", as_scitype="Panel")
        return Xt
