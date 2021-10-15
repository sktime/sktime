#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of DateTimeFeatures functionality."""

import pytest

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.date import DateTimeFeatures

# Load multivariate dataset longley and apply calendar extraction
# Test that comprehensive feature_scope works for weeks

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

# Load multivariate dataset longley and apply calendar extraction

pipe = DateTimeFeatures(base_frequency="W", feature_scope="comprehensive")

pipe.fit(Z=X_train)

# If manual_selection is provided, feature_scope should be ignored

test_complex_hi = pipe.transform(Z=X_train).columns.to_list()


pipe = DateTimeFeatures(base_frequency="W", feature_scope="minimal")

pipe.fit(Z=X_train)

# Test that minimal feature_scope works for weeks

test_complex_low = pipe.transform(Z=X_train).columns.to_list()

pipe = DateTimeFeatures(base_frequency="M", feature_scope="comprehensive")

pipe.fit(Z=X_train)

# Test that comprehensive feature_scope works for months

test_diff_basefreq = pipe.transform(Z=X_train).columns.to_list()

pipe = DateTimeFeatures(
    base_frequency="M",
    feature_scope="comprehensive",
    manual_selection=["year_of_year", "second_of_minute"],
)

pipe.fit(Z=X_train)

# Test that manual_selection works for multiple periods
# Warning should be raised since second is lower than month

test_manoverride_withbf = pipe.transform(Z=X_train).columns.to_list()

pipe = DateTimeFeatures(manual_selection=["year_of_year", "second_of_minute"])

pipe.fit(Z=X_train)

# Test that manual_selection works when nothing else is provided

test_manoverride_nobf = pipe.transform(Z=X_train).columns.to_list()

y = load_airline()

# Univariate dataset is loaded

y_train, y_test = temporal_train_test_split(y)

pipe = DateTimeFeatures(manual_selection=["year_of_year", "second_of_minute"])

pipe.fit(Z=y_train)

# Test that prior test also works for univariate data

test_diffdataset = pipe.transform(Z=y_train).columns.to_list()

y.index = y.index.to_timestamp().astype("datetime64[ns]")
y_train, y_test = temporal_train_test_split(y)

pipe = DateTimeFeatures(manual_selection=["year_of_year", "second_of_minute"])
pipe.fit(Z=y_train)

test_diffdateformat = pipe.transform(Z=y_train).columns.to_list()
# Test that prior test  also works with DateTime index


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            test_complex_hi,
            [
                "GNPDEFL",
                "GNP",
                "UNEMP",
                "ARMED",
                "POP",
                "year",
                "quarter",
                "month",
                "week_of_year",
                "month_of_quarter",
                "week_of_quarter",
                "week_of_month",
            ],
        ),
        (
            test_complex_low,
            ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "month"],
        ),
        (
            test_diff_basefreq,
            [
                "GNPDEFL",
                "GNP",
                "UNEMP",
                "ARMED",
                "POP",
                "year",
                "quarter",
                "month",
                "month_of_quarter",
            ],
        ),
        (test_manoverride_withbf, ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year"]),
        (
            test_manoverride_nobf,
            ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "second"],
        ),
        (test_diffdataset, ["Number of airline passengers", "year", "second"]),
        (test_diffdateformat, ["Number of airline passengers", "year", "second"]),
    ],
)
def test_eval(test_input, expected):
    """Tests assertions."""
    assert len(test_input) == len(expected)
    assert all([a == b for a, b in zip(test_input, expected)])
