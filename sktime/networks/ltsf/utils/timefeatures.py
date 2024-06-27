import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# used for embed (embed, fixed-embed) temporal_encoding_type
fn_month_embed = lambda x: x.month
fn_day_embed = lambda x: x.day
fn_weekday_embed = lambda x: x.weekday
fn_hour_embed = lambda x: x.hour


# used for non-embed (linear) temporal_encoding_type
fn_second = lambda x: x.second / 59.0 - 0.5
fn_minute = lambda x: x.minute / 59.0 - 0.5
fn_hour = lambda x: x.hour / 23.0 - 0.5
fn_dayofweek = lambda x: x.dayofweek / 6.0 - 0.5
fn_day = lambda x: (x.day - 1) / 30.0 - 0.5
fn_dayofyear = lambda x: (x.dayofyear - 1) / 365.0 - 0.5
fn_month = lambda x: (x.month - 1) / 11.0 - 0.5
fn_week = lambda x: (x.isocalendar().week - 1) / 52.0 - 0.5

features_by_offsets = {
    offsets.QuarterEnd: [fn_month],
    offsets.MonthEnd: [fn_month],
    offsets.Week: [fn_day, fn_week],
    offsets.Day: [fn_dayofweek, fn_day, fn_dayofyear],
    offsets.BusinessDay: [fn_dayofweek, fn_day, fn_dayofyear],
    offsets.Hour: [fn_hour, fn_dayofweek, fn_day, fn_dayofyear],
    offsets.Minute: [
        fn_minute,
        fn_hour,
        fn_dayofweek,
        fn_day,
        fn_dayofyear,
    ],
    offsets.Second: [
        fn_second,
        fn_minute,
        fn_hour,
        fn_dayofweek,
        fn_day,
        fn_dayofyear,
    ],
}


def get_mapping_functions(temporal_encoding_type, freq):

    if temporal_encoding_type == "embed" or temporal_encoding_type == "fixed-embed":
        return [
            fn_month_embed,
            fn_day_embed,
            fn_weekday_embed,
            fn_hour_embed,
        ]

    freq_offset_class = to_offset(freq).__class__
    mapping_functions = features_by_offsets.get(freq_offset_class)

    if mapping_functions:
        return mapping_functions

    # TODO: fill this
    raise ValueError()


def generate_temporal_features(index, temporal_encoding_type, freq):

    # Get mapping functions
    mapping_functions = get_mapping_functions(
        temporal_encoding_type=temporal_encoding_type,
        freq=freq,
    )

    if isinstance(index, pd.DatetimeIndex):
        index = index.to_period()
    elif not isinstance(index, pd.PeriodIndex):
        # TODO: fill this
        raise ValueError()

    index = index.map(
        lambda row: [fn(row) for fn in mapping_functions]
    )
    index = np.vstack(index)
    return index


def get_mark_vocab_sizes(temporal_encoding_type, freq):
    if temporal_encoding_type == "embed" or temporal_encoding_type == "fixed-embed":
        return [13, 32, 7, 24]
    else:
        mapping_functions = get_mapping_functions(
            temporal_encoding_type=temporal_encoding_type,
            freq=freq,
        )
        return [0] * len(mapping_functions)
