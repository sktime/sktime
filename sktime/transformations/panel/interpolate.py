# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate

from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X


class TSInterpolator(_PanelToPanelTransformer):
    """Transformer that rescales series for another number of points.
    For each cell in dataframe transformer fits scipy linear interp1d
    and samples user defined number of points. Points are generated
    by numpy.linspace. After transformation each cell will be a numpy.array
    of defined size.
    """

    _required_parameter = ["length"]

    def __init__(self, length):
        """
        Parameters
        ----------
        length : integer, the length of time series to resize to.
        """
        if length <= 0 or (not isinstance(length, int)):
            raise ValueError("resizing length must be integer and > 0")

        self.length = length
        super(TSInterpolator, self).__init__()

    def _resize_cell(self, cell):
        """Resizes the array. Firstly 1d linear interpolation is fitted on
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
        return f(np.linspace(0, 1, self.length))

    def _resize_col(self, coll):
        """Resizes column cell-wise.

        Parameters
        ----------
        coll : pandas.Series : a column with array-like objects in each cell

        Returns
        -------
        pandas.Series : a column with numpy.array in each cell with user
                        defined size
        """
        return coll.apply(self._resize_cell)

    def transform(self, X, y=None):
        """Takes series in each cell, train linear interpolation and samples n.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        pandas DataFrame : Transformed pandas DataFrame with same number
                            of rows and columns
        """
        self.check_is_fitted()
        X = check_X(X, coerce_to_pandas=True)
        return X.apply(self._resize_col)
