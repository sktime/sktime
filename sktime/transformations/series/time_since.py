# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""A transformer to compute the time elapsed since a reference time."""
from __future__ import annotations

__author__ = ["KishManani"]

import datetime
import warnings
from string import digits

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class TimeSince(BaseTransformer):
    """Computes element-wise time elapsed between the time index and a reference start time.

    Creates a column(s) which represents: `t` - `start`, where `start` is
    a reference time and `t` is the time index. The type of `start` must be
    compatible with the index of `X` used in `.fit()` and `.transform()`.

    The output can be converted to an integer representing the number of periods
    elapsed since the start time by setting `to_numeric=True`. The period is
    determined by the frequency of the index. For example, if the `freq` of
    the index is "MS" or "M" then the output is the integer number of months
    between `t` and `start`.

    Parameters
    ----------
    start : a list of start times, optional, default=None (use earliest time in index)
        a "start time" can be one of the following types:
        int - Start time to compute the time elapsed, use when index is integer
        time-like: `Period` or `datetime`
            Start time to compute the time elapsed.
        str - String is converted to datetime/period using pd.to_datetime()/pd.Period()
              to give the start time.
    to_numeric : string, optional (default=True)
        Return the integer number of periods elapsed since `start`; the period
        is defined by the frequency of the data. Converts datetime types to
        pd.Period before calculating time differences.
    freq : 'str', optional, default=None
        Only used when X has a pd.DatetimeIndex without a specified frequency.
        Specifies the frequency of the index of your data. The string should
        match a pandas offset alias:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    keep_original_columns :  boolean, optional, default=False
        Keep original columns in X passed to `.transform()`
    positive_only :  boolean, optional, default=False
        Clips negative values to zero when `to_numeric` is True.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.time_since import TimeSince
    >>> X = load_airline()

    Create a single column with time elapsed since start date of time series.
    The output is in units of integer number of months, same as the index `freq`.
    >>> transformer = TimeSince()
    >>> Xt = transformer.fit_transform(X)

    Create multiple columns with different start times. The output is in units
    of integer number of months, same as the index `freq`.
    >>> transformer = TimeSince(["2000-01", "2000-02"])
    >>> Xt = transformer.fit_transform(X)
    """

    _tags = {
        # what is the scitype of X: Series, or Panel
        "scitype:transform-input": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "univariate-only": False,
        "requires_y": False,
        "remember_data": False,
        "fit_is_empty": False,  # is fit empty and can be skipped?
        "X-y-must-have-same-index": False,
        "enforce_index_type": [pd.PeriodIndex, pd.DatetimeIndex],
        "transform-returns-same-time-index": True,
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        "capability:unequal_length:removes": False,
        "handles-missing-data": True,  # can estimator handle missing data?
        "capability:missing_values:removes": False,
    }

    def __init__(
        self,
        start: list[str | datetime.datetime | pd.Period | None] | None = None,
        *,
        to_numeric: bool | None = True,
        freq: str | None = None,
        keep_original_columns: bool | None = False,
        positive_only: bool | None = False,
    ):
        self.start = start
        self.to_numeric = to_numeric
        self.freq = freq
        self.keep_original_columns = keep_original_columns
        self.positive_only = positive_only
        super(TimeSince, self).__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        if X.index.is_numeric():
            if self.freq:
                warnings.warn("Index is integer type. `freq` will be ignored.")
            self.freq_ = None
        elif isinstance(X.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            # Chooses first non None value
            self.freq_ = X.index.freq or self.freq or pd.infer_freq(X.index)
            if self.freq_ is None:
                raise ValueError("X has no known frequency and none is supplied")
            if self.freq_ == X.index.freq and self.freq_ != self.freq and self.freq:
                warnings.warn(
                    f"Using frequency from index: {X.index.freq}, which "
                    f"does not match the frequency given: {self.freq}."
                )
        else:
            raise ValueError("Index must be of type int, datetime, or period.")

        self.start_ = []
        if self.start is None:
            self.start_.append(X.index.min())
        else:
            for start in self.start:
                if start is None:
                    self.start_.append(X.index.min())
                elif isinstance(start, str):
                    if isinstance(X.index, pd.PeriodIndex):
                        self.start_.append(pd.Period(start))
                    else:
                        self.start_.append(pd.to_datetime(start))
                else:
                    self.start_.append(start)

        # Check `start_` is compatible with index
        for start_ in self.start_:
            if isinstance(X.index, pd.DatetimeIndex) and not isinstance(
                start_, datetime.datetime
            ):
                raise ValueError(
                    f"start_={start_} incompatible with a "
                    f"datetime index. Check that `start` is of type "
                    f"datetime or a pd.Datetime parsable string."
                )
            elif isinstance(X.index, pd.PeriodIndex) and not isinstance(
                start_, pd.Period
            ):
                raise ValueError(
                    f"start_={start_} incompatible with a "
                    f"Period index. Check that `start` is of type "
                    f"pd.Period or a pd.Period parsable string."
                )
            elif X.index.is_numeric() and not isinstance(start_, (int, np.integer)):
                raise ValueError(
                    f"start_={start_} incompatible with a numeric index."
                    f"Check that `start` is an integer."
                )

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        X = X.copy()
        col_names = []
        for start_ in self.start_:
            if self.to_numeric:
                if isinstance(X.index, pd.DatetimeIndex):
                    #  To support calculating integer time differences when `freq`
                    #  is not compatible with timedelta type (e.g., "M", "MS", "Y")
                    #  X.index and `start` are first converted to pandas Period

                    # Infer the freq needed to convert to period
                    if X.index.freq is None:
                        freq_period = X.asfreq(self.freq_).index.to_period().freqstr
                    else:
                        freq_period = X.index.to_period().freqstr

                    if pd.__version__ < "1.5.0":
                        # Earlier versions of pandas returned incorrect result
                        # when taking a difference between Periods when the frequency
                        # is a multiple of a unit (e.g. "15T"). Solution is to
                        # cast to lowest frequency when taking difference (e.g., "T").
                        freq_period = _remove_digits_from_str(freq_period)

                    # Convert `start` and datetime index to period
                    start_period = pd.Period(start_, freq=freq_period)
                    X_idx_period = X.index.to_period(freq_period)
                    time_deltas_period = X_idx_period - start_period
                    time_deltas = _period_to_int(time_deltas_period)
                elif isinstance(X.index, pd.PeriodIndex):
                    if pd.__version__ < "1.5.0":
                        # Earlier versions of pandas returned incorrect result
                        # when taking a difference between Periods when the frequency
                        # is a multiple of a unit (e.g. "15T"). Solution is to
                        # cast to lowest frequency when taking difference (e.g., "T").
                        freq_ = _remove_digits_from_str(X.index.freqstr)
                        time_deltas_period = X.index.to_timestamp().to_period(
                            freq_
                        ) - start_.to_timestamp().to_period(freq_)
                    else:
                        time_deltas_period = X.index - start_
                    time_deltas = _period_to_int(time_deltas_period)
                elif X.index.is_numeric():
                    time_deltas = X.index - start_
            else:
                time_deltas = X.index - start_

            col_name = f"time_since_{start_}"
            X[col_name] = time_deltas
            col_names.append(col_name)

        if self.to_numeric and self.positive_only:
            X.loc[:, col_names] = X.loc[:, col_names].clip(lower=0)

        if self.keep_original_columns is False:
            X = X[col_names]

        return X

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {"start": None, "to_numeric": True},
            {
                "start": ["2000-01-01", "2000-01-02"],
                "to_numeric": False,
                "positive_only": True,
            },
        ]


def _period_to_int(x: pd.PeriodIndex | list[pd.offsets.DateOffset]) -> int:
    return x.map(lambda y: y.n)


def _remove_digits_from_str(x: str) -> str:
    return x.translate({ord(k): None for k in digits})
