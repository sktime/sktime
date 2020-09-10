#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from collections.abc import Iterable

import numpy as np
import pandas as pd


def timedeltas_to_int(values, freq=None):
    if not isinstance(values, Iterable):
        values = pd.TimedeltaIndex([values])
    assert isinstance(values[0], pd.Timedelta)
    if freq is None:
        if hasattr(values, "inferred_freq"):
            freq = values.inferred_freq
        else:
            raise ValueError(
                "`freq` could not be inferred automatically and no `freq` "
                "was passed.")
    return (values / pd.Timedelta(1, freq)).astype(np.int)


def date_offsets_to_int(values):
    if not isinstance(values, Iterable):
        values = [values]
    assert isinstance(values[0], pd.tseries.offsets.DateOffset)
    return pd.Int64Index([value.n for value in values])


def _subtract_time(a, b):
    """Helper function to subtract time points"""
    assert isinstance(b, (pd.Timestamp, pd.Period, int, np.integer))

    if isinstance(b, (int, np.integer)) and hasattr(a, "freq"):
        b *= a.freq

    diff = a - b

    if isinstance(b, pd.Period):
        return date_offsets_to_int(diff)
    elif isinstance(b, pd.Timestamp):
        if hasattr(a, "freq") and a.freqstr is not None:
            freq = a.freqstr
        elif hasattr(b, "freq") and b.freqstr is not None:
            freq = b.freqstr
        else:
            raise ValueError("No `freq` information available")
        return timedeltas_to_int(diff, freq=freq)
    else:
        return diff
