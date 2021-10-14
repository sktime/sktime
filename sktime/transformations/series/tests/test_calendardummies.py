#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of CalendarDummies functionality."""


import pandas as pd
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.calendardummies import CalendarDummies

# Load multivariate dataset longley and apply calendar extraction
# First test:

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

# Load multivariate dataset longley and apply calendar extraction

pipe = CalendarDummies(base_frequency="week", complexity=2)

pipe.fit(Z=X_train)

# If manual_selection is provided, complexity should be ignored

first_test = pipe.transform(Z=X_train).columns.to_list()


pipe = CalendarDummies(base_frequency="week", complexity=0)

pipe.fit(Z=X_train)

# Test that complexity 0 works for weeks

second_test = pipe.transform(Z=X_train).columns.to_list()

pipe = CalendarDummies(base_frequency="month", complexity=2)

pipe.fit(Z=X_train)

# Test that complexity 2 works for months

third_test = pipe.transform(Z=X_train).columns.to_list()

pipe = CalendarDummies(
    base_frequency="month",
    complexity=2,
    manual_selection=["year_of_year", "second_of_minute"],
)

pipe.fit(Z=X_train)

# Test that manual_selection works for multiple periods
# Warning should be raised since second is lower than month

fourth_test = pipe.transform(Z=X_train).columns.to_list()

pipe = CalendarDummies(manual_selection=["year_of_year", "second_of_minute"])

pipe.fit(Z=X_train)

# Test that manual_selection works when nothing else is provided

fifth_test = pipe.transform(Z=X_train).columns.to_list()

y, X = load_airline(), pd.DataFrame(index=load_airline().index)

# Univariate dataset is loaded

y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

pipe = CalendarDummies(manual_selection=["year_of_year", "second_of_minute"])

pipe.fit(Z=X_train)

# Test that test five also works for univariate data

sixth_test = pipe.transform(Z=X_train).columns.to_list()

airline = load_airline()
airline.index = airline.index.to_timestamp().astype("datetime64[ns]")
y, X = airline, pd.DataFrame(index=airline.index)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

pipe = CalendarDummies(manual_selection=["year_of_year", "second_of_minute"])
pipe.fit(Z=X_train)

seventh_test = pipe.transform(Z=X_train).columns.to_list()
# Test that test five also works with DateTime index


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            first_test,
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
        (second_test, ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "month"]),
        (
            third_test,
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
        (fourth_test, ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year"]),
        (fifth_test, ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "second"]),
        (sixth_test, ["year", "second"]),
    ],
)
def test_eval(test_input, expected):
    """Tests assertions."""
    assert len(test_input) == len(expected)
    assert all([a == b for a, b in zip(test_input, expected)])
