#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of DateTimeFeatures functionality."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.date import DateTimeFeatures

# Load multivariate dataset longley and apply calendar extraction

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

# Test that comprehensive feature_scope works for weeks
pipe = DateTimeFeatures(ts_freq="W", feature_scope="comprehensive")
pipe.fit(X_train)
test_full_featurescope = pipe.transform(X_train).columns.to_list()

# Test that minimal feature_scope works for weeks
pipe = DateTimeFeatures(ts_freq="W", feature_scope="minimal")
pipe.fit(X_train)
test_reduced_featurescope = pipe.transform(X_train).columns.to_list()

# Test that comprehensive feature_scope works for months
pipe = DateTimeFeatures(ts_freq="M", feature_scope="comprehensive")
pipe.fit(X_train)
test_changing_frequency = pipe.transform(X_train).columns.to_list()

# Test that manual_selection works for with provided arguments
# Should ignore feature scope and raise warning for second_of_minute,
# since ts_freq = "M" is provided.
# (dummies with frequency higher than ts_freq)
pipe = DateTimeFeatures(
    ts_freq="M",
    feature_scope="comprehensive",
    manual_selection=["year", "second_of_minute"],
)
pipe.fit(X_train)
test_manspec_with_tsfreq = pipe.transform(X_train).columns.to_list()

# Test that manual_selection works for with provided arguments
# Should ignore feature scope and raise no warning for second_of_minute,
# since ts_freq is not provided.

pipe = DateTimeFeatures(manual_selection=["year", "second_of_minute"])
pipe.fit(X_train)
test_manspec_wo_tsfreq = pipe.transform(X_train).columns.to_list()

# Test that prior test works for with univariate dataset
y = load_airline()
y_train, y_test = temporal_train_test_split(y)

pipe = DateTimeFeatures(manual_selection=["year", "second_of_minute"])
pipe.fit(y_train)
test_univariate_data = pipe.transform(y_train).columns.to_list()

# Test that prior test also works when Index is converted to DateTime index
y.index = y.index.to_timestamp().astype("datetime64[ns]")
y_train, y_test = temporal_train_test_split(y)
pipe = DateTimeFeatures(manual_selection=["year", "second_of_minute"])
pipe.fit(y_train)
test_diffdateformat = pipe.transform(y_train).columns.to_list()

pipe = DateTimeFeatures(ts_freq="L", feature_scope="comprehensive")
pipe.fit(y_train)
test_full = pipe.transform(y_train).columns.to_list()
test_types = pipe.transform(y_train).select_dtypes(include=["int64"]).columns.to_list()


# Test `is_weekend` works in manual selection
@pytest.fixture
def df_datetime_daily_idx():
    """Create timeseries with Datetime index, daily frequency."""
    return pd.DataFrame(
        data={"y": [1, 1, 1, 1, 1, 1, 1]},
        index=pd.date_range(start="2000-01-01", freq="D", periods=7),
    )


all_args = [
    "Number of airline passengers",
    "year",
    "quarter_of_year",
    "month_of_year",
    "week_of_year",
    "day_of_year",
    "month_of_quarter",
    "week_of_quarter",
    "day_of_quarter",
    "week_of_month",
    "day_of_month",
    "day_of_week",
    "hour_of_day",
    "minute_of_hour",
    "second_of_minute",
    "millisecond_of_second",
    "is_weekend",
]


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            test_full_featurescope,
            [
                "GNPDEFL",
                "GNP",
                "UNEMP",
                "ARMED",
                "POP",
                "year",
                "quarter_of_year",
                "month_of_year",
                "week_of_year",
                "month_of_quarter",
                "week_of_quarter",
                "week_of_month",
            ],
        ),
        (
            test_reduced_featurescope,
            ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "month_of_year"],
        ),
        (
            test_changing_frequency,
            [
                "GNPDEFL",
                "GNP",
                "UNEMP",
                "ARMED",
                "POP",
                "year",
                "quarter_of_year",
                "month_of_year",
                "month_of_quarter",
            ],
        ),
        (
            test_manspec_with_tsfreq,
            ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "second_of_minute"],
        ),
        (
            test_manspec_wo_tsfreq,
            ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "second_of_minute"],
        ),
        (
            test_univariate_data,
            ["Number of airline passengers", "year", "second_of_minute"],
        ),
        (
            test_diffdateformat,
            ["Number of airline passengers", "year", "second_of_minute"],
        ),
        (
            test_full,
            all_args,
        ),
        (
            test_types,
            all_args[1:],
        ),
    ],
)
def test_eval(test_input, expected):
    """Tests which columns are returned for different arguments.

    For a detailed description what these arguments do,
    and how they interact see docstring of DateTimeFeatures.
    """
    assert len(test_input) == len(expected)
    assert all([a == b for a, b in zip(test_input, expected)])


def test_manual_selection_is_weekend(df_datetime_daily_idx):
    """Tests that "is_weekend" returns correct result in `manual_selection`."""
    transformer = DateTimeFeatures(manual_selection=["is_weekend"])

    Xt = transformer.fit_transform(df_datetime_daily_idx)
    expected = pd.DataFrame(
        data={"y": [1, 1, 1, 1, 1, 1, 1], "is_weekend": [1, 1, 0, 0, 0, 0, 0]},
        index=df_datetime_daily_idx.index,
    )
    assert_frame_equal(Xt, expected)
