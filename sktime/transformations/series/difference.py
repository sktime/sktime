#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Class to iteratively apply differences to a time series."""
__author__ = ["Ryan Kuhns"]
__all__ = ["Differencer"]

import numpy as np
import pandas as pd
from sklearn.utils import check_array

from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation import is_int
from sktime.utils.validation.series import check_series


def _check_lags(lags):
    msg = " ".join(
        [
            "`lags` should be provided as a positive integer scaler, or",
            "a list, tuple or np.ndarray of positive integers,"
            f"but found {type(lags)}.",
        ]
    )
    non_positive_msg = "`lags` should be positive integers."
    if isinstance(lags, int):
        if lags <= 0:
            raise ValueError(non_positive_msg)
        lags = check_array([lags], ensure_2d=False)
    elif isinstance(lags, (list, tuple, np.ndarray)):
        if not all([is_int(lag) for lag in lags]):
            raise TypeError(msg)
        lags = check_array(lags, ensure_2d=False)
        if (lags <= 0).any():
            raise ValueError(non_positive_msg)
    else:
        raise TypeError(msg)

    return lags


def _inverse_diff(series, lag):
    for i in range(lag):
        series.iloc[i::lag] = series.iloc[i::lag].cumsum()

    return series


class Differencer(_SeriesToSeriesTransformer):
    """Apply iterative differences to a timeseries.

    Difference transformations are applied at the specified lags in the order
    provided. For example, using lags=[1, 12] corresponds to applying a
    standard first difference, then differencing the first-differenced series
    at lag 12 (in the event the input data has a monthly periodicity, this
    would equate to a first difference followed by a seasonal difference).

    The transformation works for univariate and multivariate timeseries. However,
    the multivariate case applies the same differencing to every series.

    Parameters
    ----------
    lags : int or array-like, default = 1
        The lags used to difference the data.
        If a single `int` value is

    remove_missing : bool, default = True
        Whether the differencer should remove the initial observations that
        contain missing values as a result of the differencing operation(s).

    Attributes
    ----------
    lags : np.ndarray
        Lags used to perform the differencing of the input series.

    remove_missing : bool
        Stores whether the Differencer removes initial observations that contain
        missing values as a result of the differencing operation(s).

    Example
    -------
    >>> from sktime.transformations.series.difference import Differencer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Differencer(lags=[1, 12])
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "fit-in-transform": False,
        "transform-returns-same-time-index": False,
        "univariate-only": False,
    }

    def __init__(self, lags=1, remove_missing=True):
        self.lags = lags
        self.remove_missing = remove_missing
        self._Z = None
        self._cumulative_lags = None
        self._prior_cum_lags = None
        self._prior_lags = None
        super(Differencer, self).__init__()

    def _check_inverse_transform_index(self, Z):
        """Check fitted series contains indices needed in inverse_transform."""
        first_idx = Z.index.min()
        orig_first_idx, orig_last_idx = self._Z.index.min(), self._Z.index.max()

        is_contained_by_fitted_z = False
        is_future = False

        if first_idx < orig_first_idx:
            msg = [
                "Some indices of `Z` are prior to timeseries used in `fit`.",
                "Reconstruction via `inverse_transform` is not possible.",
            ]
            raise ValueError(" ".join(msg))

        elif Z.index.difference(self._Z.index).shape[0] == 0:
            is_contained_by_fitted_z = True

        elif first_idx > orig_last_idx:
            is_future = True

        pad_z_inv = self.remove_missing or is_future

        cutoff = Z.index[0] if pad_z_inv else Z.index[self._cumulative_lags[-1]]
        fh = ForecastingHorizon(
            np.array([*range(-1, -(self._cumulative_lags[-1] + 1), -1)])
        )

        index = fh.to_absolute(cutoff).to_pandas()
        index_diff = index.difference(self._Z.index)

        if index_diff.shape[0] != 0:
            msg = [
                f"Inverse transform requires indices {index}",
                "to have been stored in `fit()`,",
                f"but the indices {index_diff} were not found.",
            ]
            raise ValueError(" ".join(msg))

        return is_contained_by_fitted_z, pad_z_inv

    def _fit(self, Z, X=None):
        """Logic used by fit method on `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the specified difference transformation on.

        Returns
        -------
        self
        """
        self.lags = _check_lags(self.lags)
        self._prior_lags = np.roll(self.lags, shift=1)
        self._prior_lags[0] = 0
        self._cumulative_lags = self.lags.cumsum()
        self._prior_cum_lags = np.zeros_like(self._cumulative_lags)
        self._prior_cum_lags[1:] = self._cumulative_lags[:-1]
        self._Z = Z.copy()
        return self

    def _transform(self, Z, lags):
        """Logic used by `transform` to apply transformation to `Z`.

        Differences are applied at lags specified in `lags`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            The timeseries to be differenced.

        lags : np.ndarray
            Lags to be used in applying differences.

        Returns
        -------
        diff :
            Differenced series.
        """
        diff = Z.copy()

        if len(lags) == 0:
            return diff

        else:
            for lag in lags:
                diff = diff.diff(lag)

            return diff

    def _inverse_transform(self, Z, X=None):
        """Logic used by `inverse_transform` to reverse transformation on  `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the specified difference transformation on.

        Returns
        -------
        Z_inv : pd.Series or pd.DataFrame
            The reconstructed timeseries after the transformation has been reversed.
        """
        is_df = isinstance(Z, pd.DataFrame)
        (
            is_contained_by_fitted_z,
            pad_z_inv,
        ) = self._check_inverse_transform_index(Z)

        # If `Z` is entirely contained in fitted `_Z` we can just return
        # the values from the timeseires stored in `fit` as a shortcut
        if is_contained_by_fitted_z:
            Z_inv = self._Z.loc[Z.index, :] if is_df else self._Z.loc[Z.index]

        else:
            Z_inv = Z.copy()
            for i, lag_info in enumerate(
                zip(self.lags[::-1], self._prior_cum_lags[::-1])
            ):
                lag, prior_cum_lag = lag_info
                _lags = self.lags[::-1][i + 1 :]
                _transformed = self._transform(self._Z, _lags)

                # Determine index values for initial values needed to reverse
                # the differencing for the specified lag
                if pad_z_inv:
                    cutoff = Z_inv.index[0]
                else:
                    cutoff = Z_inv.index[prior_cum_lag + lag]
                fh = ForecastingHorizon(np.array([*range(-1, -(lag + 1), -1)]))
                index = fh.to_absolute(cutoff).to_pandas()

                if is_df:
                    prior_n_timepoint_values = _transformed.loc[index, :]
                else:
                    prior_n_timepoint_values = _transformed.loc[index]
                if pad_z_inv:
                    Z_inv = pd.concat([prior_n_timepoint_values, Z_inv])
                else:
                    Z_inv.update(prior_n_timepoint_values)

                Z_inv = _inverse_diff(Z_inv, lag)

        if pad_z_inv:
            Z_inv = Z_inv.loc[Z.index, :] if is_df else Z_inv.loc[Z.index]

        return Z_inv

    def fit(self, Z, X=None):
        """Fit the transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the specified difference transformation on.

        Returns
        -------
        self
        """
        Z = check_series(Z)

        self._fit(Z, X=X)

        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Return transformed version of input series `Z`.

        Difference transformations are applied to a series iteratively.
        Differences are applied at lags specified in `lags`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the specified difference transformation on.

        Returns
        -------
        Z_transform : pd.Series
            Transformed version of input series `Z`.
        """
        self.check_is_fitted()
        Z = check_series(Z)

        Z_transform = self._transform(Z, self.lags)

        if self.remove_missing:
            Z_transform = Z_transform.iloc[self._cumulative_lags[-1] :]

        return Z_transform

    def inverse_transform(self, Z, X=None):
        """Reverse transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the specified difference transformation on.

        Returns
        -------
        Z_inv : pd.Series or pd.DataFrame
            The reconstructed timeseries after the transformation has been reversed.
        """
        self.check_is_fitted()
        Z = check_series(Z)

        Z_inv = self._inverse_transform(Z, X=X)

        return Z_inv
