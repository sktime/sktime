"""Discrete wavelet transform."""

import numpy as np
import pandas as pd

from sktime.datatypes import convert
from sktime.transformations.base import BaseTransformer

__author__ = ["vnicholson1"]


class DWTTransformer(BaseTransformer):
    """Discrete Wavelet Transform Transformer.

    Performs the Haar wavelet transformation on a time series.

    Parameters
    ----------
    num_levels : int, number of levels to perform the Haar wavelet
        transformation.

    Examples
    --------
    >>> from sktime.transformations.dwt import DWTTransformer
    >>> from sktime.datasets import load_airline
    >>> from sktime.datatypes import convert
    >>>
    >>> y = load_airline()
    >>> transformer = DWTTransformer(num_levels=3)
    >>> y_transformed = transformer.fit_transform(y)
    """

    _tags = {
        "authors": "vnicholson1",
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
        "capability:categorical_in_X": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, num_levels=3):
        self.num_levels = num_levels
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series
            transformed version of X
        """
        self._check_parameters()

        # Get information about the dataframe
        col_names = X.columns

        Xt = pd.DataFrame()
        for x in col_names:
            # Convert one of the columns in the dataframe to numpy array
            arr = convert(
                pd.DataFrame(X[x]),
                from_type="nested_univ",
                to_type="numpyflat",
                as_scitype="Panel",
            )

            transformedData = self._extract_wavelet_coefficients(arr)

            # Convert to a numpy array
            transformedData = np.asarray(transformedData)

            # Add it to the dataframe
            colToAdd = []
            for i in range(len(transformedData)):
                inst = transformedData[i]
                colToAdd.append(pd.Series(inst))

            Xt[x] = colToAdd

        return Xt

    def _extract_wavelet_coefficients(self, data):
        """Extract wavelet coefficients of a 2d array of time series.

        The coefficients correspond to the approximation coefficients of the highest
        level, followed by the wavelet coefficients from levels num_levels to 1.
        """
        data = np.asarray(data)
        if self.num_levels == 0:
            return data

        wav_coeffs = []
        approx = data
        for _ in range(self.num_levels):
            # a length-1 series is its own approximation and wavelet coefficient
            if approx.shape[1] == 1:
                wav_coeffs.append(approx)
                continue
            # pairwise Haar step on all instances at once, odd last value dropped
            n_pairs = approx.shape[1] // 2
            even = approx[:, 0 : 2 * n_pairs : 2]
            odd = approx[:, 1 : 2 * n_pairs : 2]
            wav_coeffs.append((even - odd) / np.sqrt(2))
            approx = (even + odd) / np.sqrt(2)

        return np.hstack([approx] + wav_coeffs[::-1])

    def _check_parameters(self):
        """Check the values of parameters passed to DWT.

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Provides two parameter sets so the estimator is covered by the
        `test_get_test_params_coverage` test (issue #3429).
        """
        # default / simple parameter sets
        params1 = {"num_levels": 3}
        params2 = {"num_levels": 100}
        # alternate parameter set: edge case with zero levels
        params3 = {"num_levels": 0}

        return [params1, params2, params3]
