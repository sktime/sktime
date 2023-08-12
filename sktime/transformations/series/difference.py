#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Class to iteratively apply differences to a time series."""
__author__ = ["RNKuhns", "fkiraly"]
__all__ = ["Differencer"]

from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_array

from sktime.datatypes._utilities import get_cutoff, update_data
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


def _diff_transform(X: Union[pd.Series, pd.DataFrame], lags: np.array):
    """Perform differencing on Series or DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
    lags : int or iterable of int, e.g., list of int

    Returns
    -------
    `X` differenced at lags `lags`, always a copy (no reference)
    if `lags` is int, applies diff to X at period `lags`
        returns X.diff(periods=lag)
    if `lags` is list of int, loops over elements from start to end
        and applies diff to X at period lags[value], for value in the list `lags`
    """
    if isinstance(lags, int):
        lags = [lags]

    Xt = X

    for lag in lags:
        # converting lag to int since pandas complains if it's np.int64
        Xt = Xt.diff(periods=int(lag))

    return Xt


def _diff_to_seq(X: Union[pd.Series, pd.DataFrame], lags: np.array):
    """Difference a series multiple times and return intermediate results.

    Parameters
    ----------
    X : pd.DataFrame
    lags : int or iterable of int, e.g., list of int

    Returns
    -------
    list, i-th element is _diff_transform(X, lags[0:i])
    """
    if X is None:
        return None

    if isinstance(lags, int):
        lags = [lags]

    ret = [X]
    Xd = X
    for lag in lags:
        # converting lag to int since pandas complains if it's np.int64
        Xd = Xd.diff(periods=int(lag))
        ret += [Xd]
    return ret


def _shift(ix, periods):
    """Shift pandas index by periods."""
    if isinstance(ix, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        return ix.shift(periods)
    else:
        return ix + periods


def _inverse_diff(X, lags, X_diff_seq=None):
    """Inverse to difference.

    Parameters
    ----------
    X : pd.Series or pd.DataFrame
    lags : int or iterable of int, e.g., list of int
    X_diff_seq : list of pd.Series or pd.DataFrame
        elements must match type, columns and index type of X
        length must be equal or longer than length of lags

    Returns
    -------
    `X` inverse differenced at lags `lags`, always a copy (no reference)
    if `lags` is int, applies cumsum to X at period `lag`
        for i in range(lag), X.iloc[i::lag] = X.iloc[i::lag].cumsum()
    if `lags` is list of int, loops over elements from start to end
        and applies cumsum to X at period lag[value], for value in the list `lag`
    if `X_diff_seq` is provided, uses values stored for indices outside `X` to invert
    """
    if isinstance(lags, int):
        lags = [lags]

    # if lag is numpy, convert to list
    if isinstance(lags, (np.ndarray, list, tuple)):
        lags = list(lags)

    # if lag is a list, recurse
    if isinstance(lags, (list, tuple)):
        if len(lags) == 0:
            return X

    lags = lags.copy()

    # lag_first = pop last element of lags
    lag_last = lags.pop()

    # invert last lag index
    if X_diff_seq is not None:
        # Get the train time series before the last difference
        X_diff_orig = X_diff_seq[len(lags)]
        # Shift the differenced time series index by the last lag
        # to match the original time series index
        X_ix_shift = _shift(X.index, -lag_last)
        # Get the original time series values for the intersecting
        # indices between the shifted index and the original index
        X_update = X_diff_orig.loc[X_ix_shift.intersection(X_diff_orig.index)]
        # Set the values of the differenced time series to nan for all indices
        # that are in the indices of the original and the by the sum of all lags
        # shifted original time series that are available in the differenced time
        # series (intersection). These are the indices for which no valid differenced
        # values exist.
        X.loc[
            X_diff_orig.index.difference(
                _shift(X_diff_orig.index, sum(lags) + lag_last)
            ).intersection(X.index)
        ] = np.nan
        X = X.combine_first(X_update)

    X_diff_last = X.copy()

    if lag_last < 0:
        X_diff_last = X_diff_last.iloc[::-1]

    abs_lag = abs(lag_last)

    for i in range(abs_lag):
        X_diff_last.iloc[i::abs_lag] = X_diff_last.iloc[i::abs_lag].cumsum()

    if lag_last < 0:
        X_diff_last = X_diff_last.iloc[::-1]

    # if any more lags, recurse
    if len(lags) > 0:
        return _inverse_diff(X_diff_last, lags, X_diff_seq=X_diff_seq)
    # else return
    else:
        return X_diff_last


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

    na_handling : str, optional, default = "fill_zero"
        How to handle the NaNs that appear at the start of the series from differencing
        Example: there are only 3 differences in a series of length 4,
            differencing [a, b, c, d] gives [?, b-a, c-b, d-c]
            so we need to determine what happens with the "?" (= unknown value)
        "drop_na" - unknown value(s) are dropped, the series is shortened
        "keep_na" - unknown value(s) is/are replaced by NaN
        "fill_zero" - unknown value(s) is/are replaced by zero

    memory : str, optional, default = "all"
        how much of previously seen X to remember, for exact reconstruction of inverse
        "all" : estimator remembers all X, inverse is correct for all indices seen
        "latest" : estimator only remembers latest X necessary for future reconstruction
            inverses at any time stamps after fit are correct, but not past time stamps
        "none" : estimator does not remember any X, inverse is direct cumsum

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

    def __init__(self, lags=1, na_handling="fill_zero", memory="all"):
        self.lags = lags
        self.na_handling = self._check_na_handling(na_handling)
        self.memory = memory

        self._X = None
        self._lags = _check_lags(self.lags)
        self._cumulative_lags = None
        super().__init__()

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

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

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
        memory = self.memory

        lagsum = self._lags.cumsum()[-1]
        self._lagsum = lagsum

        # remember X or part of X
        if memory == "all":
            self._X = X
        elif memory == "latest":
            n_memory = min(len(X), lagsum)
            self._X = X.iloc[-n_memory:]

        self._freq = get_cutoff(X, return_index=True)
        return self

    def _check_freq(self, X):
        """Ensure X carries same freq as X seen in _fit."""
        if self._freq is not None and hasattr(self._freq, "freq"):
            if hasattr(X.index, "freq") and X.index.freq is None:
                X.index.freq = self._freq.freq
        return X

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
        X_orig_index = X.index

        X = update_data(X=self._X, X_new=X)

        X = self._check_freq(X)

        Xt = _diff_transform(X, self._lags)

        Xt = Xt.loc[X_orig_index]

        na_handling = self.na_handling
        if na_handling == "drop_na":
            Xt = Xt.iloc[self._lagsum :]
        elif na_handling == "fill_zero":
            Xt.iloc[: self._lagsum] = 0
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
        lags = self._lags

        X_diff_seq = _diff_to_seq(self._X, lags)

        X = self._check_freq(X)

        X_orig_index = X.index

        Xt = _inverse_diff(X, lags, X_diff_seq=X_diff_seq)

        Xt = Xt.loc[X_orig_index]

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
