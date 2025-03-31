#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformers for detecting outliers in a time series."""

__author__ = ["aiwalter"]
__all__ = ["HampelFilter"]

import warnings
from math import ceil

import numpy as np
import pandas as pd

from sktime.split import SlidingWindowSplitter
from sktime.transformations.base import BaseTransformer


class HampelFilter(BaseTransformer):
    """Use HampelFilter to detect outliers based on a sliding window.

    Correction of outliers is recommended by means of the sktime.Imputer,
    so both can be tuned separately.

    Parameters
    ----------
    window_length : int, optional (default=10)
        Length of the sliding window
    n_sigma : int, optional (default=3)
        Defines how strong a point must outly to be an "outlier"
    k : float, optional (default = 1.4826)
        A constant scale factor which is dependent on the distribution,
        for Gaussian it is approximately 1.4826, by default 1.4826
    return_bool : bool, optional (default=False)
        If True, outliers are filled with True and non-outliers with False.
        Else, outliers are filled with np.nan.

    Notes
    -----
    Implementation is based on [1]_.

    References
    ----------
    .. [1] Hampel F. R., "The influence curve and its role in robust estimation",
       Journal of the American Statistical Association, 69, 382-393, 1974

    Examples
    --------
    >>> from sktime.transformations.series.outlier_detection import HampelFilter
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = HampelFilter(window_length=10)
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "authors": ["aiwalter"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": True,
        "handles-missing-data": True,
        "skip-inverse-transform": True,
        "univariate-only": False,
    }

    def __init__(self, window_length=10, n_sigma=3, k=1.4826, return_bool=False):
        self.window_length = window_length
        self.n_sigma = n_sigma
        self.k = k
        self.return_bool = return_bool
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        Z = X.copy()

        # multivariate
        if isinstance(Z, pd.DataFrame):
            for col in Z:
                Z[col] = self._transform_series(Z[col])
        # univariate
        else:
            Z = self._transform_series(Z)

        Xt = Z
        return Xt

    def _transform_series(self, Z):
        """Logic internal to the algorithm for transforming the input series.

        Parameters
        ----------
        Z : pd.Series

        Returns
        -------
        pd.Series
        """
        # warn if nan values in Series, as user might mix them
        # up with outliers otherwise
        if Z.isnull().values.any():
            warnings.warn(
                """Series contains nan values, more nan might be
                added if there are outliers""",
                stacklevel=2,
            )

        cv = SlidingWindowSplitter(
            fh=0,
            window_length=self.window_length,
            step_length=1,
            start_with_window=True,
        )
        half_window_length = int(self.window_length / 2)

        Z = _hampel_filter(
            Z=Z,
            cv=cv,
            n_sigma=self.n_sigma,
            half_window_length=half_window_length,
            k=self.k,
        )

        # data post-processing
        if self.return_bool:
            Z = Z.apply(lambda x: bool(np.isnan(x)))

        return Z

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        param1 = {"window_length": 3}
        param2 = {}
        param3 = {"window_length": 5, "n_sigma": 2, "k": 1.7, "return_bool": True}
        return [param1, param2, param3]


def _hampel_filter(Z, cv, n_sigma, half_window_length, k):
    for i in cv.split(Z):
        cv_window = i[0]
        cv_median = np.nanmedian(Z.iloc[cv_window])
        cv_sigma = k * np.nanmedian(np.abs(Z.iloc[cv_window] - cv_median))

        is_start_window = cv_window[-1] == cv.window_length - 1
        is_end_window = cv_window[-1] == len(Z) - 1
        if is_start_window:
            idx_range = range(cv_window[0], half_window_length + 1)
        elif is_end_window:
            if cv.window_length % 2 == 0:
                start_end_win = half_window_length
            else:
                start_end_win = ceil(cv.window_length / 2)
            idx_range = range(len(Z) - start_end_win, len(Z))
        else:
            idx_range = [cv_window[0] + half_window_length]

        for idx in idx_range:
            loc_idx = Z.index[idx]  # convert to loc to avoid write on copy
            Z.loc[loc_idx] = _compare(
                value=Z.iloc[idx],
                cv_median=cv_median,
                cv_sigma=cv_sigma,
                n_sigma=n_sigma,
            )
    return Z


def _compare(value, cv_median, cv_sigma, n_sigma):
    """Identify an outlier.

    Parameters
    ----------
    value : int/float
    cv_median : int/float
    cv_sigma : int/float
    n_sigma : int/float

    Returns
    -------
    int/float or np.nan
        Returns value if value it is not an outlier,
        else np.nan (or True/False if return_bool==True)
    """
    if np.abs(value - cv_median) > n_sigma * cv_sigma:
        return np.nan
    else:
        return value
