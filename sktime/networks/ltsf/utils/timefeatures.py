"""Generates Features for Temporal Emboddings."""

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# used for embed (embed, fixed-embed) temporal_encoding_type


def fn_month_embed(x):
    """Extract the month from a timestamp."""
    return x.month


def fn_day_embed(x):
    """Extract the day from a timestamp."""
    return x.day


def fn_weekday_embed(x):
    """Extract the weekday from a timestamp."""
    return x.weekday


def fn_hour_embed(x):
    """Extract the hour from a timestamp."""
    return x.hour


# used for non-embed (linear) temporal_encoding_type


def fn_second(x):
    """Normalize the second from a timestamp."""
    return x.second / 59.0 - 0.5


def fn_minute(x):
    """Normalize the minute from a timestamp."""
    return x.minute / 59.0 - 0.5


def fn_hour(x):
    """Normalize the hour from a timestamp."""
    return x.hour / 23.0 - 0.5


def fn_dayofweek(x):
    """Normalize the day of the week from a timestamp."""
    return x.dayofweek / 6.0 - 0.5


def fn_day(x):
    """Normalize the day from a timestamp."""
    return (x.day - 1) / 30.0 - 0.5


def fn_dayofyear(x):
    """Normalize the day of the year from a timestamp."""
    return (x.dayofyear - 1) / 365.0 - 0.5


def fn_month(x):
    """Normalize the month from a timestamp."""
    return (x.month - 1) / 11.0 - 0.5


def fn_week(x):
    """Normalize the week of the year from a timestamp."""
    return (x.isocalendar().week - 1) / 52.0 - 0.5


features_by_offsets = {
    offsets.QuarterEnd: [fn_month],
    offsets.MonthEnd: [fn_month],
    offsets.Week: [fn_day, fn_week],
    offsets.Day: [fn_dayofweek, fn_day, fn_dayofyear],
    offsets.BusinessDay: [fn_dayofweek, fn_day, fn_dayofyear],
    offsets.Hour: [fn_hour, fn_dayofweek, fn_day, fn_dayofyear],
    offsets.Minute: [fn_minute, fn_hour, fn_dayofweek, fn_day, fn_dayofyear],
    offsets.Second: [fn_second, fn_minute, fn_hour, fn_dayofweek, fn_day, fn_dayofyear],
}


def get_mapping_functions(temporal_encoding_type, freq):
    """Get the mapping functions based on the temporal encoding type and frequency."""
    if temporal_encoding_type in {"embed", "fixed-embed"}:
        return [fn_month_embed, fn_day_embed, fn_weekday_embed, fn_hour_embed]

    freq_offset_class = to_offset(freq).__class__
    mapping_functions = features_by_offsets.get(freq_offset_class)

    if mapping_functions is None:
        # no specific frequency found
        mapping_functions = [
            fn_month_embed,
            fn_day_embed,
            fn_weekday_embed,
            fn_hour_embed,
        ]

    return mapping_functions


def generate_temporal_features(index, temporal_encoding_type, freq):
    """Generate temporal features for a given index, encoding type, and frequency."""
    mapping_functions = get_mapping_functions(temporal_encoding_type, freq)

    if isinstance(index, pd.DatetimeIndex):
        index = index.to_period()

    index = index.map(lambda row: [fn(row) for fn in mapping_functions])
    index = np.vstack(index)
    return index


def get_mark_vocab_sizes(temporal_encoding_type, freq):
    """Get vocabulary sizes for mark embeddings based on the encoding type and freq."""
    if temporal_encoding_type in {"embed", "fixed-embed"}:
        return [13, 32, 7, 24]
    else:
        mapping_functions = get_mapping_functions(temporal_encoding_type, freq)
        return [0] * len(mapping_functions)
