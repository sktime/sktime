# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.utils.validation.panel import check_X

__author__ = "Vincent Nicholson"


class DWTTransformer(_PanelToPanelTransformer):

    """
    The Discrete Wavelet Transform Transformer. This class performs
    the Haar wavelet transformation on a time series.

    Parameters
    ----------
    num_levels : int, number of levels to perform the Haar wavelet
                 transformation.
    """

    def __init__(self, num_levels=3):
        self.num_levels = num_levels
        super(DWTTransformer, self).__init__()

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : a pandas dataframe of shape = [n_samples, num_dims]
            The training input samples.

        Returns
        -------
        dims: a pandas data frame of shape
              = [n_samples, num_dims]
        """

        # Check the data
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False, coerce_to_pandas=True)

        self._check_parameters()

        # Get information about the dataframe
        col_names = X.columns

        df = pd.DataFrame()
        for x in col_names:
            # Convert one of the columns in the dataframe to numpy array
            arr = from_nested_to_2d_array(pd.DataFrame(X[x]), return_numpy=True)

            transformedData = self._extract_wavelet_coefficients(arr)

            # Convert to a numpy array
            transformedData = np.asarray(transformedData)

            # Add it to the dataframe
            colToAdd = []
            for i in range(len(transformedData)):
                inst = transformedData[i]
                colToAdd.append(pd.Series(inst))

            df[x] = colToAdd

        return df

    def _extract_wavelet_coefficients(self, data):
        """
        Function to extract the wavelet coefficients
        of a 2d array of time series.

        The coefficients correspond to the wavelet coefficients
        from levels 1 to num_levels followed by the approximation
        coefficients of the highest level.
        """
        num_levels = self.num_levels
        res = []

        for x in data:
            if num_levels == 0:
                res.append(x)
            else:
                coeffs = []
                current = x
                approx = None
                for _ in range(num_levels):
                    approx = self._get_approx_coefficients(current)
                    wav_coeffs = self._get_wavelet_coefficients(current)
                    current = approx
                    wav_coeffs.reverse()
                    coeffs.extend(wav_coeffs)
                approx.reverse()
                coeffs.extend(approx)
                coeffs.reverse()
                res.append(coeffs)

        return res

    def _check_parameters(self):
        """
        Function for checking the values of parameters inserted into DWT.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.num_levels, int):
            if self.num_levels <= -1:
                raise ValueError("num_levels must have the value" + "of at least 0")
        else:
            raise TypeError(
                "num_levels must be an 'int'. Found"
                + "'"
                + type(self.num_levels).__name__
                + "' instead."
            )

    def _get_approx_coefficients(self, arr):
        """
        Function to get the approximate coefficients at a given level.
        """
        new = []
        if len(arr) == 1:
            return [arr[0]]
        for x in range(math.floor(len(arr) / 2)):
            new.append((arr[2 * x] + arr[2 * x + 1]) / math.sqrt(2))
        return new

    def _get_wavelet_coefficients(self, arr):
        """
        Function to get the wavelet coefficients at a given level.
        """
        new = []
        # if length is 1, just return the list back
        if len(arr) == 1:
            return [arr[0]]
        for x in range(math.floor(len(arr) / 2)):
            new.append((arr[2 * x] - arr[2 * x + 1]) / math.sqrt(2))
        return new
