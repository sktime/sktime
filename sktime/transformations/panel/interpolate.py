# -*- coding: utf-8 -*-
"""Time series interpolator/re-sampler."""
import numpy as np
import pandas as pd
from scipy import interpolate

from sktime.transformations.base import BaseTransformer

__author__ = ["mloning"]


class TSInterpolator(BaseTransformer):
    """Time series interpolator/re-sampler.

    Transformer that rescales series for another number of points.
    For each cell in dataframe transformer fits scipy linear interp1d
    and samples user defined number of points. Points are generated
    by numpy.linspace.

    After transformation each cell will be a pd.Series of given length.
    Indices of the pd.Series will be changed to integer indices.

    Parameters
    ----------
    length : integer, the length of time series to resize to.
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
    }

    def __init__(self, length):
        """Initialize estimator.

        Parameters
        ----------
        length : integer, the length of time series to resize to.
        """
        if length <= 0 or (not isinstance(length, int)):
            raise ValueError("resizing length must be integer and > 0")

        self.length = length
        super(TSInterpolator, self).__init__()

    def _resize_cell(self, cell):
        """Resize a single array.

        Resizes the array. Firstly 1d linear interpolation is fitted on
           original array as y and numpy.linspace(0, 1, len(cell)) as x.
           Then user defined number of points is sampled in
           numpy.linspace(0, 1, length) and returned into cell as numpy array.

        Parameters
        ----------
        cell : array-like

        Returns
        -------
        numpy.array : with user defined size
        """
        f = interpolate.interp1d(list(np.linspace(0, 1, len(cell))), cell.to_numpy())
        Xt = f(np.linspace(0, 1, self.length))
        return pd.Series(Xt)

    def _transform(self, X, y=None):
        """Take series in each cell, train linear interpolation and samples n.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time series in cells, following nested_univ format.
        y : ignored argument for interface compatibility

        Returns
        -------
        pandas DataFrame : Transformed pandas DataFrame of shape [n_samples, n_features]
            follows nested_univ format
        """
        return X.applymap(self._resize_cell)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        params = {"length": 10}
        return params
