#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = []

import re

import numpy as np
import pandas as pd

from sktime.utils.validation.series import check_time_index


def _coerce_duration_to_int(duration, freq=None):
    """Coerce durations into integer representations for a given unit of
    duration

    Parameters
    ----------
    duration : pd.DateOffset, pd.Timedelta, pd.TimedeltaIndex, pd.Index, int
        Duration type or collection of duration types
    freq : str
        Frequency

    Returns
    -------
    ret : int
        Duration in integer values for given unit
    """
    if isinstance(duration, pd.tseries.offsets.DateOffset):
        return duration.n
    elif isinstance(duration, pd.Index) and isinstance(
        duration[0], pd.tseries.offsets.BaseOffset
    ):
        return pd.Int64Index([d.n for d in duration])
    elif isinstance(duration, (pd.Timedelta, pd.TimedeltaIndex)):
        if freq is None:
            raise ValueError("`unit` missing")
        # Supports eg: W, 3W, W-SUN, BQS, (B)Q(S)-MAR patterns, from which we
        # extract the count and the unit. See
        # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        m = re.match(r"(?P<count>\d*)(?P<unit>[a-zA-Z]+)$", freq)
        if not m:
            raise ValueError("pandas frequency %s not understood." % freq)
        count, unit = m.groups()
        count = 1 if not count else int(count)
        # integer conversion only works reliably with non-ambiguous units (
        # e.g. days, seconds but not months, years)
        try:
            if isinstance(duration, pd.Timedelta):
                return int(duration / pd.Timedelta(count, unit))
            if isinstance(duration, pd.TimedeltaIndex):
                return (duration / pd.Timedelta(count, unit)).astype(np.int)
        except ValueError:
            raise ValueError(
                "Index type not supported. Please consider using pd.PeriodIndex."
            )
    else:
        raise TypeError("`duration` type not understood.")


def _get_freq(x):
    """Get unit for conversion of time deltas to integers"""
    if hasattr(x, "freqstr"):
        return x.freqstr
    else:
        return None


def _shift(x, by=1):
    """Shift time point `x` by a step (`by`) given frequency of `x`

    Parameters
    ----------
    x : pd.Period, pd.Timestamp, int
        Time point
    by : int

    Returns
    -------
    ret : pd.Period, pd.Timestamp, int
        Shifted time point
    """
    assert isinstance(x, (pd.Period, pd.Timestamp, int, np.integer)), type(x)
    assert isinstance(by, (int, np.integer, pd.Int64Index)), type(by)
    if isinstance(x, pd.Timestamp):
        if not hasattr(x, "freq") or x.freq is None:
            raise ValueError("No `freq` information available")
        by *= x.freq
    return x + by


def _get_duration(x, y=None, coerce_to_int=False, unit=None):
    """Compute duration of time index `x` or durations between time
    points `x` and `y` if `y` is given

    Parameters
    ----------
    x : pd.Index, pd.Timestamp, pd.Period, int
    y : pd.Timestamp, pd.Period, int, optional (default=None)
    coerce_to_int : bool
        If True, duration is returned as integer value for given unit
    unit : str
        Time unit

    Returns
    -------
    ret : duration type
        Duration
    """
    if y is None:
        x = check_time_index(x)
        duration = x[-1] - x[0]
    else:
        assert isinstance(x, (int, np.integer, pd.Period, pd.Timestamp))
        # check types allowing (np.integer, int) combinations to pass
        assert type(x) is type(y) or (
            isinstance(x, (np.integer, int)) and isinstance(x, (np.integer, int))
        )
        duration = x - y

    # coerce to integer result for given time unit
    if coerce_to_int and isinstance(
        x, (pd.PeriodIndex, pd.DatetimeIndex, pd.Period, pd.Timestamp)
    ):
        if unit is None:
            # try to get the unit from the data if not given
            unit = _get_freq(x)
        duration = _coerce_duration_to_int(duration, freq=unit)
    return duration
