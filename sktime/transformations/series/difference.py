#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Class for apply iterative differences to a time series."""
__author__ = ["Ryan Kuhns"]
__all__ = ["Differencer"]

import math

import numpy as np
import pandas as pd
from sklearn.utils import check_array

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation import is_int
from sktime.utils.validation.series import check_series


def _check_differencer_lags(lags):
    msg = " ".join(
        [
            "`lags` should be provided as a integer scaler, or",
            f"a list, tuple or np.ndarray of integers, but found {type(lags)}",
        ]
    )
    if isinstance(lags, int):
        lags = check_array([lags], ensure_2d=False)
    elif isinstance(lags, (list, tuple, np.ndarray)):
        if not all([is_int(lag) for lag in lags]):
            raise TypeError(msg)
        lags = check_array(lags, ensure_2d=False)
    else:
        raise TypeError(msg)

    return lags


def _difference_info(series, lag, prior_cum_lags):
    return {
        "first_n_timepoints": series[prior_cum_lags : prior_cum_lags + lag],
        "last_n_timepoints": series[-lag:],
    }


def _inverse_diff(series, start, stop, lag, first_n_timepoints, n_timepoints):
    def _inverse_diff_slice_values(series, start, stop, lag):
        return series[start:stop].values + series[start - lag : stop - lag].values

    if lag == 1:
        series.iloc[start:stop] = first_n_timepoints.values
        series = series.cumsum()
    else:
        iterations = math.ceil((n_timepoints - start) / lag)
        for it in range(iterations):
            if it == 0:
                series.iloc[start:stop] = first_n_timepoints.values
            else:
                start = np.clip(start + lag, 0, n_timepoints)
                stop = np.clip(stop + lag, 0, n_timepoints)
                series.iloc[start:stop] = _inverse_diff_slice_values(
                    series, start, stop, lag
                )

    return series


class Differencer(_SeriesToSeriesTransformer):
    """Difference series.

    Parameters
    ----------
    lags : int or array-like, default = 1
        The lags used to difference the data.
        If a single `int` value is

    use_with_predict : bool, default = False
        Whether the Differencer is being used to inverses-transform predictions.
        This affects how the `inverse_transform` method reverses the transformation.

    Attributes
    ----------
    lags : np.ndarray
        Lags used to perform the differencing of the input series.

    use_with_predict : bool
        Stores whether the Differencer is being used to inverse-tranform predictions.

    Example
    -------
    >>> from sktime.transformations.series.difference import Differencer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Differencer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "fit-in-transform": True,
        "transform-returns-same-time-index": True,
        "univariate-only": True,
    }

    def __init__(self, lags=1, use_with_predict=False):
        self.lags = _check_differencer_lags(lags)
        self.use_with_predict = use_with_predict
        self._cumalative_lags = self.lags.cumsum()

        self._prior_cum_lags = np.zeros_like(self._cumalative_lags)
        self._prior_cum_lags[1:] = self._cumalative_lags[:-1]

        self._lag_fit_info = None
        super(Differencer, self).__init__()

    def _fit_difference(self, series, lags):
        """Apply a differences to a series iteratively.

        Differences are applied at lags specified in `lags`.

        Parameters
        ----------
        series : pd.Series
            The series to be differenced.

        lags : np.ndarray
            Lags to be used in applying differences.

        Returns
        -------
        diff :
            Differenced series.
        """
        diff = series.copy()
        for i, lag in enumerate(self.lags):
            prior_cum_lag = self._prior_cum_lags[i]
            self._lag_fit_info.append(_difference_info(diff, lag, prior_cum_lag))
            diff = diff.diff(lag)

        return diff

    def transform(self, Z, X=None):
        """Return transformed version of input series `Z`.

        Parameters
        ----------
        Z : pd.Series
            A time series to apply the specified difference transformation on.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        self._lag_fit_info = []

        return self._fit_difference(Z, self.lags)

    def inverse_transform(self, Z, X=None):
        """Reverse transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.Series
            A time series to apply the specified difference transformation on.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        Z_inv = Z.copy()

        n_timepoints = Z_inv.shape[0]

        for lag, fit, prior_cum_lag in zip(
            self.lags[::-1], self._lag_fit_info[::-1], self._prior_cum_lags[::-1]
        ):
            if self.use_with_predict:
                start = 0
                first_n_timepoints = fit["last_n_timepoints"]
                Z_inv = pd.concat([pd.Series([np.nan] * lag), Z_inv])
                n_timepoints = Z_inv.shape[0]
            else:
                start = prior_cum_lag
                first_n_timepoints = fit["first_n_timepoints"]

            stop = start + lag

            Z_inv = _inverse_diff(
                Z_inv, start, stop, lag, first_n_timepoints, n_timepoints
            )
            if self.use_with_predict:
                Z_inv = Z_inv.iloc[lag:]

        return Z_inv
