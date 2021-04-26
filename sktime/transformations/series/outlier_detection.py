#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Martin Walter"]
__all__ = ["HampelFilter"]

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sktime.forecasting.model_selection import SlidingWindowSplitter

import numpy as np
import warnings


class HampelFilter(_SeriesToSeriesTransformer):
    """HampelFilter to detect outliers based on a sliding window. Correction
    of outliers is recommended by means of the sktime.Imputer,
    so both can be tuned separately.

    Parameters
    ----------
    window_length : int, optional (default=10)
        Lenght of the sliding window
    n_sigma : int, optional
        Defines how strong a point must outly to be an "outlier", by default 3
    k : float, optional
        A constant scale factor which is dependent on the distribution,
        for Gaussian it is approximately 1.4826, by default 1.4826
    return_bool : bool, optional
        If True, outliers are filled with True and non-outliers with False.
        Else, outliers are filled with np.nan.

    References
    ----------
    Hampel F. R., "The influence curve and its role in robust estimation",
    Journal of the American Statistical Association, 69, 382–393, 1974

    Example
    ----------
    >>> from sktime.transformations.series.outlier_detection import HampelFilter
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = HampelFilter(window_length=10)
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "univariate-only": True,
        "fit-in-transform": True,
        "handles-missing-data": True,
        "skip-inverse-transform": True,
    }

    def __init__(self, window_length=10, n_sigma=3, k=1.4826, return_bool=False):

        self.window_length = window_length
        self.n_sigma = n_sigma
        self.k = k
        self.return_bool = return_bool
        super(HampelFilter, self).__init__()

    def transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)

        # warn if nan values in Series, as user might mix them
        # up with outliers otherwise
        if z.isnull().values.any():
            warnings.warn(
                """Series contains nan values, more nan might be
                added if there are outliers"""
            )

        cv = SlidingWindowSplitter(
            window_length=self.window_length, step_length=1, start_with_window=True
        )
        half_window_length = int(self.window_length / 2)

        z = _hampel_filter(
            z=z,
            cv=cv,
            n_sigma=self.n_sigma,
            half_window_length=half_window_length,
            k=self.k,
        )

        # data post-processing
        if self.return_bool:
            z = z.apply(lambda x: True if np.isnan(x) else False)

        return z


def _hampel_filter(z, cv, n_sigma, half_window_length, k):
    for i in cv.split(z):
        cv_window = i[0]
        cv_median = np.nanmedian(z[cv_window])
        cv_sigma = k * np.nanmedian(np.abs(z[cv_window] - cv_median))

        # find outliers at start and end of z
        if (
            cv_window[0] <= half_window_length
            or cv_window[-1] >= len(z) - half_window_length
        ) and (cv_window[0] in [0, len(z) - cv.window_length - 1]):

            # first half of the first window
            if cv_window[0] <= half_window_length:
                idx_range = range(cv_window[0], half_window_length + 1)

            # last half of the last window
            else:
                idx_range = range(len(z) - half_window_length - 1, len(z))
            for j in idx_range:
                z.iloc[j] = _compare(
                    value=z.iloc[j],
                    cv_median=cv_median,
                    cv_sigma=cv_sigma,
                    n_sigma=n_sigma,
                )
        else:
            idx = cv_window[0] + half_window_length
            z.iloc[idx] = _compare(
                value=z.iloc[idx],
                cv_median=cv_median,
                cv_sigma=cv_sigma,
                n_sigma=n_sigma,
            )
    return z


def _compare(value, cv_median, cv_sigma, n_sigma):
    """Function to identify an outlier

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
