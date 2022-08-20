#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Class to iteratively apply differences to a time series."""
__author__ = ["RNKuhns", "fkiraly"]
__all__ = ["Differencer"]

from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_array

from sktime.datatypes._utilities import get_cutoff
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.base import BaseTransformer
from sktime.utils.validation import is_int


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


class Differencer(BaseTransformer):
    """Apply iterative differences to a timeseries.

    The transformation works for univariate and multivariate timeseries. However,
    the multivariate case applies the same differencing to every series.

    Difference transformations are applied at the specified lags in the order provided.

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

    na_handling : str, default = "fill_zero"
        How to handle the NaNs that appear at the start of the series from differencing
        Example: there are only 3 differences in a series of length 4,
            differencing [a, b, c, d] gives [?, b-a, c-b, d-c]
            so we need to determine what happens with the "?" (= unknown value)
        "drop_na" - unknown value(s) are dropped, the series is shortened
        "keep_na" - unknown value(s) is/are replaced by NaN
        "fill_zero" - unknown value(s) is/are replaced by zero

    Examples
    --------
    >>> from sktime.transformations.series.difference import Differencer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Differencer(lags=[1, 12])
    >>> y_transform = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,
        "transform-returns-same-time-index": False,
        "univariate-only": False,
        "capability:inverse_transform": True,
    }

    VALID_NA_HANDLING_STR = ["drop_na", "keep_na", "fill_zero"]

    def __init__(self, lags=1, na_handling="fill_zero"):
        self.lags = lags
        self.na_handling = self._check_na_handling(na_handling)

        self._Z = None
        self._lags = None
        self._cumulative_lags = None
        self._prior_cum_lags = None
        self._prior_lags = None
        super(Differencer, self).__init__()

        # if the na_handling is "fill_zero" or "keep_na"
        #   then the returned indices are same to the passed indices
        if self.na_handling in ["fill_zero", "keep_na"]:
            self.set_tags(**{"transform-returns-same-time-index": True})

    def _check_na_handling(self, na_handling):
        """Check na_handling parameter, should be a valid string as per docstring."""
        if na_handling not in self.VALID_NA_HANDLING_STR:
            raise ValueError(
                f'invalid na_handling parameter value encountered: "{na_handling}", '
                f"na_handling must be one of: {self.VALID_NA_HANDLING_STR}"
            )

        return na_handling

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

        pad_z_inv = self.na_handling == "drop_na" or is_future

        cutoff = Z.index[0] if pad_z_inv else Z.index[self._cumulative_lags[-1]]
        fh = ForecastingHorizon(
            np.arange(-1, -(self._cumulative_lags[-1] + 1), -1), freq=self._freq
        )
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

    def _fit(self, X, y=None):
        """
        Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        self._lags = _check_lags(self.lags)
        self._prior_lags = np.roll(self._lags, shift=1)
        self._prior_lags[0] = 0
        self._cumulative_lags = self._lags.cumsum()
        self._prior_cum_lags = np.zeros_like(self._cumulative_lags)
        self._prior_cum_lags[1:] = self._cumulative_lags[:-1]
        self._Z = X.copy()

        self._freq = get_cutoff(X, return_index=True)
        return self

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
        Xt = _diff_transform(X, self._lags)

        na_handling = self.na_handling
        if na_handling == "drop_na":
            Xt = Xt.iloc[self._cumulative_lags[-1] :]
        elif na_handling == "fill_zero":
            Xt.iloc[: self._cumulative_lags[-1]] = 0
        elif na_handling == "keep_na":
            pass
        else:
            raise RuntimeError(
                "unreachable condition, invalid na_handling value encountered: "
                f"{na_handling}"
            )
        return Xt

    def _inverse_transform(self, X, y=None):
        """Logic used by `inverse_transform` to reverse transformation on `X`.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be inverse transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            inverse transformed version of X
        """
        Z = X
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
                fh = ForecastingHorizon(np.arange(-1, -(lag + 1), -1), freq=self._freq)
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

        Xt = Z_inv

        return Xt

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [{"na_handling": x} for x in cls.VALID_NA_HANDLING_STR]
        # we're testing that inverse_transform is inverse to transform
        #   and that is only correct if the first observation is not dropped
        # todo: ensure that we have proper tests or escapes for "incomplete inverses"
        params = params[1:]
        #   this removes "drop_na" setting where the inverse has problems
        #   need to deal with this in a better way in testing
        return params
