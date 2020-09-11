__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd


def _coerce_duration_to_int(duration, unit=None):
    """Coerce durations into integer representations for a given unit of duration"""
    if isinstance(duration, pd.tseries.offsets.DateOffset):
        return duration.n
    elif isinstance(duration, pd.Index) and \
            isinstance(duration[0], pd.tseries.offsets.BaseOffset):
        return pd.Int64Index([d.n for d in duration])
    elif isinstance(duration, (pd.Timedelta, pd.TimedeltaIndex)):
        if unit is None:
            raise ValueError("`unit` missing")
        try:
            if isinstance(duration, pd.Timedelta):
                return int(duration / pd.Timedelta(1, unit))
            if isinstance(duration, pd.TimedeltaIndex):
                return (duration / pd.Timedelta(1, unit)).astype(np.int)
        except ValueError:
            raise ValueError(
                "Index type not supported. Please consider using pd.PeriodIndex.")
    else:
        raise TypeError("`duration` type not understood.")


def _get_unit(x):
    """Get unit for conversion of time deltas to integers"""
    if hasattr(x, "freqstr"):
        return x.freqstr
    else:
        return None


def _shift(x, by=1):
    """Shift time point `x`"""
    if isinstance(x, pd.Timestamp):
        if not hasattr(x, "freq"):
            raise ValueError("no `freq` information available")
        by *= x.freq
    return x + by


def _get_duration(x, y=None, coerce_to_int=False, unit=None):
    """Compute duration of time index `x`"""
    if y is None:
        assert isinstance(x, pd.Index)
        duration = x[-1] - x[0]
    else:
        assert isinstance(x, (int, np.integer, pd.Period, pd.Timestamp))
        assert type(x) is type(y) or (isinstance(x, (np.integer, int)) and
                                      isinstance(x, (np.integer, int)))
        duration = x - y
    if coerce_to_int and isinstance(x, (pd.PeriodIndex, pd.DatetimeIndex, pd.Period,
                                        pd.Timestamp)):
        if unit is None:
            unit = _get_unit(x)
        duration = _coerce_duration_to_int(duration, unit=unit)
    return duration
