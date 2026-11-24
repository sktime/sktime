"""Discrete wavelet transform."""

import math

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__author__ = ["vnicholson1", "shubham5080"]


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
    >>>
    >>> y = load_airline()
    >>> transformer = DWTTransformer(num_levels=3)
    >>> y_transformed = transformer.fit_transform(y)
    """

    _tags = {
        "authors": ["vnicholson1", "shubham5080"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd-multiindex",  # replace nested_univ (#10716)
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
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
        X : pd.DataFrame with MultiIndex (instance, time)
            Panel of equal-length series; columns are variables
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.DataFrame with MultiIndex (instance, coefficient)
            Haar wavelet coefficients per variable
        """
        self._check_parameters()

        instances = X.index.get_level_values(0).unique()

        frames = []
        for col in X.columns:
            # shape (n_instances, n_timepoints)
            arr = np.stack(
                [X.xs(i, level=0)[col].to_numpy(dtype=float) for i in instances]
            )
            transformed = np.asarray(
                self._extract_wavelet_coefficients(arr), dtype=float
            )
            n_coeffs = transformed.shape[1]
            idx = pd.MultiIndex.from_product(
                [instances, range(n_coeffs)],
                names=X.index.names,
            )
            frames.append(pd.Series(transformed.ravel(), index=idx, name=col))

        return pd.concat(frames, axis=1)

    def _extract_wavelet_coefficients(self, data):
        """Extract wavelet coefficients of a 2d array of time series.

        The coefficients correspond to the wavelet coefficients from levels 1 to
        num_levels followed by the approximation coefficients of the highest level.
        """
        num_levels = self.num_levels
        res = []

        for x in data:
            if num_levels == 0:
                res.append(list(x))
            else:
                coeffs = []
                current = list(x)
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

    def _get_approx_coefficients(self, arr):
        """Get the approximate coefficients at a given level."""
        new = []
        if len(arr) == 1:
            return [arr[0]]
        for x in range(math.floor(len(arr) / 2)):
            new.append((arr[2 * x] + arr[2 * x + 1]) / math.sqrt(2))
        return new

    def _get_wavelet_coefficients(self, arr):
        """Get the wavelet coefficients at a given level."""
        new = []
        # if length is 1, just return the list back
        if len(arr) == 1:
            return [arr[0]]
        for x in range(math.floor(len(arr) / 2)):
            new.append((arr[2 * x] - arr[2 * x + 1]) / math.sqrt(2))
        return new

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
