# -*- coding: utf-8 -*-
"""Tests for holiday functionality."""

import numpy as np
import pandas as pd
import pytest
from holidays import CountryHoliday

from sktime.transformations.series.holidays import HolidayFeatures


@pytest.fixture
def y10():
    """Pytest fixture for time series for 10 years."""
    index = pd.date_range(start="2010-01-01", end="2019-12-31", freq="D")
    values = np.empty(index.shape[0])
    y = pd.Series(values, index=index)
    return y


CALENDAR = CountryHoliday("FR")


def test_holidays_return_columns(y10):
    """Test if all columns are returned."""
    transformer = HolidayFeatures(
        calendar=CALENDAR,
        return_dummies=True,
        return_categorical=True,
        return_indicator=True,
    )
    yt = transformer.fit_transform(y10)
    columns = yt.columns

    # check indicator column
    assert "is_holiday" in columns
    assert yt["is_holiday"].dtype == int

    # check categorical column
    assert "holiday" in columns
    assert yt["holiday"].dtype == "category"
    categories = yt["holiday"].cat.categories.drop("no_holiday")

    # check dummy columns
    for category in categories:
        assert category in columns
        assert yt[category].dtype == int


def test_holidays_capture_all_holidays(y10):
    """Test if all holidays are captured."""
    transformer = HolidayFeatures(
        calendar=CALENDAR, return_dummies=False, return_categorical=True
    )
    yt = transformer.fit_transform(y10)
    assert np.all(yt.groupby("holiday").count().drop("no_holiday") == 10)


def test_holidays_include_bridge_days(y10):
    """Test if bridge days are captured."""
    transformer = HolidayFeatures(
        calendar=CALENDAR,
        return_dummies=False,
        return_categorical=True,
        include_bridge_days=True,
    )
    yt = transformer.fit_transform(y10)
    assert np.all(yt.groupby("holiday").count().drop("no_holiday") >= 10)


def test_holidays_window(y10):
    """Test if window of days around holidays are captured."""
    transformer = HolidayFeatures(
        calendar=CALENDAR,
        return_dummies=False,
        return_categorical=True,
        holiday_windows={"Noël": (1, 3), "Jour de l'an": (0, 1)},
    )
    yt = transformer.fit_transform(y10)

    # We include 4 days around Christmas, so there should be a total of 50 holidays.
    assert np.all(yt.groupby("holiday").count().loc["Noël"] == 50)
    # We include 1 days around New Year, so there should be a total of 20 holidays.
    assert np.all(yt.groupby("holiday").count().loc["Jour de l'an"] == 20)
