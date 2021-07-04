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


# TODO: INCORPORATE pd.Index.intersection/pd.Index.difference() IF POSSIBLE
def _check_index_overlap(index1, index2):
    first_idx_one, last_idx_one = index1.min(), index1.max()
    first_idx_two, last_idx_two = index2.min(), index2.max()
    # Check if index1 is entirely prior to index2
    if last_idx_one < first_idx_two:
        index1_vs_index2_case = "past"
    # Check if index1 is at least partially before index2
    elif first_idx_one < first_idx_two:
        index1_vs_index2_case = "past with partial overlap"
    # Check if index1 is entirely after index2
    elif first_idx_one > last_idx_two:
        index1_vs_index2_case = "future"
    # Otherwise the input series at least partially overlaps fitted series
    elif first_idx_one >= first_idx_two and last_idx_one <= last_idx_two:
        index1_vs_index2_case = "total overlap"
    # Otherwise the index1 is partially in the future
    else:
        index1_vs_index2_case = "future with partial overlap"

    return index1_vs_index2_case


def _inverse_diff2(series, start, stop, lag, first_n_timepoints):
    series.iloc[start:stop] = first_n_timepoints.values
    for i in range(lag):
        series.iloc[i::lag] = series.iloc[i::lag].cumsum()

    return series


def _inverse_diff(series, lag):
    for i in range(lag):
        series.iloc[i::lag] = series.iloc[i::lag].cumsum()

    return series


class Differencer(_SeriesToSeriesTransformer):
    """Apply iterative differences to a timeseries.

    Difference transformations are applied at the specified lags in the order
    given. For example, using lags=[1, 12] corresponds to first appying a
    standard first difference, then differencing the first-differenced series
    at lag 12 (in the event the input data has a monthly periodicity, this
    would equate to a first difference followed by a seasonal difference).

    The transformation works for univariate and multivariate timeseries. However,
    the multivariate case applies the same differencing to every series. If

    Parameters
    ----------
    lags : int or array-like, default = 1
        The lags used to difference the data.
        If a single `int` value is

    remove_missing : bool, default = True
        Whether the differencer should remove the initial observations that
        contain missing values as a result of the differncing operation(s).

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
    >>> transformer = Differencer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "fit-in-transform": False,
        "transform-returns-same-time-index": True,
        "univariate-only": False,
        "multivariate-only": False,
    }

    def __init__(self, lags=1, remove_missing=True):
        self.lags = lags
        self.remove_missing = remove_missing
        self._Z = None
        self._cumalative_lags = None
        self._prior_cum_lags = None
        self._prior_lags = None
        super(Differencer, self).__init__()

    def _check_index_location(self, Z_inv):
        first_idx, last_idx = Z_inv.index.min(), Z_inv.index.max()

        orig_first_idx_loc = self._Z.index.get_loc(first_idx)
        orig_last_idx_loc = self._Z.index.get_loc(last_idx)
        is_future = True
        if orig_first_idx_loc < self._cumalative_lags[-1:]:
            msg = " ".join(
                [
                    "Not enough indices in fitted series prior to first index",
                    "of series to be inverse transformed",
                ]
            )
            raise ValueError(msg)
        return orig_first_idx_loc, orig_last_idx_loc, is_future

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
        self._cumalative_lags = self.lags.cumsum()
        self._prior_cum_lags = np.zeros_like(self._cumalative_lags)
        self._prior_cum_lags[1:] = self._cumalative_lags[:-1]
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
        Z_inv = Z.copy()

        is_df = isinstance(Z_inv, pd.DataFrame)

        inverse_index_case = _check_index_overlap(Z_inv.index, self._Z.index)
        pad_z_inv = self.remove_missing or inverse_index_case == "future"
        for i, lag_info in enumerate(zip(self.lags[::-1], self._prior_cum_lags[::-1])):
            lag, prior_cum_lag = lag_info
            _lags = self.lags[::-1][i + 1 :]
            _transformed = self._transform(self._Z, _lags)

            if pad_z_inv:
                cutoff = Z_inv.index[0]
            else:
                cutoff = Z_inv.index[prior_cum_lag + lag]

            fh = ForecastingHorizon(np.array([*range(-1, -(lag + 1), -1)]))
            index = fh.to_absolute(cutoff).to_pandas()
            index_diff = index.difference(self._Z.index)
            if index_diff.shape[0] != 0:
                msg = " ".join(
                    [
                        f"Inverse transform requires indices {index}",
                        "to have been stored in `fit()`,",
                        f"but the indices {index_diff} were not found.",
                    ]
                )
                raise ValueError(msg)

            else:
                if is_df:
                    prior_periods = _transformed.loc[index, :]
                else:
                    prior_periods = _transformed.loc[index]
            if pad_z_inv:
                Z_inv = pd.concat([prior_periods, Z_inv])
            else:
                Z_inv.update(prior_periods)

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
            Z_transform = Z_transform.iloc[self._cumalative_lags[-1] :]

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

        # if self.remove_missing or inverse_index_case == 'future':
        #     # start = 0
        #     fh = ForecastingHorizon(np.array([*range(-1, -(lag+1), -1)]))
        #     index = fh.to_absolute(Z_inv.index[0]).to_pandas()
        #     index_diff = index.difference(self._Z.index)
        #     if index_diff.shape[0] != 0:
        #         msg = " ".join(
        #             [
        #                 f"Inverse transform requires indices {index}",
        #                 "to have been stored in `fit()`,",
        #                 f"but the indices {index_diff} were not found."
        #             ]
        #         )
        #         raise ValueError(msg)
        #     else:
        #         if is_df:
        #             prior_periods = _transformed.loc[index, :]
        #         else:
        #             prior_periods = _transformed.loc[index]
        #     Z_inv = pd.concat([prior_periods, Z_inv])

        # else:
        #     start = prior_cum_lag

        # stop = start + lag

        # if is_df:
        #     first_n_timepoints = _transformed.iloc[
        #         prior_cum_lag : prior_cum_lag + lag, :
        #     ]
        #     Z_inv.iloc[start:stop, :] = first_n_timepoints
        # else:
        #     first_n_timepoints = _transformed[prior_cum_lag : prior_cum_lag + lag]
        #     Z_inv.iloc[start:stop] = first_n_timepoints
