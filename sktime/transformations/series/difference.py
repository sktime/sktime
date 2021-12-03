#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Class to iteratively apply differences to a time series."""
__author__ = ["Ryan Kuhns"]
__all__ = ["Differencer"]

from typing import Union

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


def _diff_transform(Z: Union[pd.Series, pd.DataFrame], lags: np.array):
    """Perform differencing on Series or DataFrame."""
    Zt = Z.copy()

    if len(lags) != 0:
        for lag in lags:
            # converting lag to int since pandas complains if it's np.int64
            Zt = Zt.diff(periods=int(lag))

    return Zt


def _inverse_diff(Z, lag):
    for i in range(lag):
        Z.iloc[i::lag] = Z.iloc[i::lag].cumsum()

    return Z


class Differencer(_SeriesToSeriesTransformer):
    """Apply iterative differences to a timeseries.

    The transformation works for univariate and multivariate timeseries. However,
    the multivariate case applies the same differencing to every series.

    Difference transformations are applied at the specified lags in the order
    provided.

    For example, given a timeseries with monthly periodicity, using lags=[1, 12]
    corresponds to applying a standard first difference to handle trend, and
    followed by a seasonal difference (at lag 12) to attempt to account for
    seasonal dependence.

    To provide a higher-order difference at the same lag list the lag multiple
    times. For example, lags=[1, 1] takes iterative first differences like may
    be needed for a series that is integrated of order 2.

    Parameters
    ----------
    lags : int or array-like, default = 1
        The lags used to difference the data.
        If a single `int` value is

    drop_na : bool, default = True
        Whether the differencer should drop the initial observations that
        contain missing values as a result of the differencing operation(s).

    Attributes
    ----------
    lags : int or array-like
        Lags used to perform the differencing of the input series.

    drop_na : bool
        Stores whether the Differencer drops the initial observations that contain
        missing values as a result of the differencing operation(s).

    Examples
    --------
    >>> from sktime.transformations.series.difference import Differencer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Differencer(lags=[1, 12])
    >>> y_transform = transformer.fit_transform(y)
    """

    _tags = {
        "fit-in-transform": False,
        "transform-returns-same-time-index": False,
        "univariate-only": False,
    }

    def __init__(self, lags=1, drop_na=True):
        self.lags = lags
        self.drop_na = drop_na
        self._Z = None
        self._lags = None
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

        pad_z_inv = self.drop_na or is_future

        cutoff = Z.index[0] if pad_z_inv else Z.index[self._cumulative_lags[-1]]
        fh = ForecastingHorizon(np.arange(-1, -(self._cumulative_lags[-1] + 1), -1))
        index = fh.to_absolute(cutoff).to_pandas()
        index_diff = index.difference(self._Z.index)

        if index_diff.shape[0] != 0 and not is_contained_by_fitted_z:
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
            A timeseries to apply the specified transformation on.

        Returns
        -------
        self
        """
        self._lags = _check_lags(self.lags)
        self._prior_lags = np.roll(self._lags, shift=1)
        self._prior_lags[0] = 0
        self._cumulative_lags = self._lags.cumsum()
        self._prior_cum_lags = np.zeros_like(self._cumulative_lags)
        self._prior_cum_lags[1:] = self._cumulative_lags[:-1]
        self._Z = Z.copy()
        return self

    def _transform(self, Z, X=None):
        """Logic used by `transform` to apply transformation to `Z`.

        Differences are applied at lags specified in `lags`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            The timeseries to apply the specified transformation on.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            The transformed timeseries.
        """
        Zt = _diff_transform(Z, self._lags)
        if self.drop_na:
            Zt = Zt.iloc[self._cumulative_lags[-1] :]
        return Zt

    def _inverse_transform(self, Z, X=None):
        """Logic used by `inverse_transform` to reverse transformation on  `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to reverse the transformation on.

        Returns
        -------
        Z_inv : pd.Series or pd.DataFrame
            The reconstructed timeseries after the transformation has been reversed.
        """
        is_df = isinstance(Z, pd.DataFrame)
        is_contained_by_fit_z, pad_z_inv = self._check_inverse_transform_index(Z)

        # If `Z` is entirely contained in fitted `_Z` we can just return
        # the values from the timeseires stored in `fit` as a shortcut
        if is_contained_by_fit_z:
            Z_inv = self._Z.loc[Z.index, :] if is_df else self._Z.loc[Z.index]

        else:
            Z_inv = Z.copy()
            for i, lag_info in enumerate(
                zip(self._lags[::-1], self._prior_cum_lags[::-1])
            ):
                lag, prior_cum_lag = lag_info
                _lags = self._lags[::-1][i + 1 :]
                _transformed = _diff_transform(self._Z, _lags)

                # Determine index values for initial values needed to reverse
                # the differencing for the specified lag
                if pad_z_inv:
                    cutoff = Z_inv.index[0]
                else:
                    cutoff = Z_inv.index[prior_cum_lag + lag]
                fh = ForecastingHorizon(np.arange(-1, -(lag + 1), -1))
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
            A time series to apply the specified transformation on.

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

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the specified transformation on.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            Transformed version of input series `Z`.
        """
        self.check_is_fitted()
        Z = check_series(Z)

        return self._transform(Z, X=X)

    def inverse_transform(self, Z, X=None):
        """Reverse transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to reverse the transformation on.

        Returns
        -------
        Z_inv : pd.Series or pd.DataFrame
            The reconstructed timeseries after the transformation has been reversed.
        """
        self.check_is_fitted()
        Z = check_series(Z)

        return self._inverse_transform(Z, X=X)
