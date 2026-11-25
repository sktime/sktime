#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for HolidayFeatures functionality."""

__author__ = ["VyomkeshVyas", "fnhirwa"]

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.holiday._holidayfeats import HolidayFeatures


@pytest.fixture
def calendar():
    """Fixture for GB holidays."""
    from holidays import country_holidays

    return country_holidays(country="GB")


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_return_dummies(calendar):
    """Tests return_dummies param."""
    X = pd.DataFrame(
        {"values": np.arange(1, 6)},
        index=pd.date_range("2022-05-01", periods=5, freq="D"),
    )
    trafo_dm = HolidayFeatures(calendar=calendar, return_dummies=True)
    X_trafo_dm = trafo_dm.fit_transform(X).astype(np.int32)
    expected_dm = pd.DataFrame({"May Day": np.int32([0, 1, 0, 0, 0])}, index=X.index)
    assert_frame_equal(X_trafo_dm, expected_dm)


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_return_categorical(calendar):
    """Tests return_categorical param."""
    X = pd.DataFrame(
        {"values": np.arange(1, 6)},
        index=pd.date_range("2022-05-01", periods=5, freq="D"),
    )
    trafo_ctg = HolidayFeatures(
        calendar=calendar, return_categorical=True, return_dummies=False
    )
    X_trafo_ctg = trafo_ctg.fit_transform(X)
    expected_ctg = pd.DataFrame(
        {
            "holiday": pd.Categorical(
                ["no_holiday", "May Day", "no_holiday", "no_holiday", "no_holiday"]
            )
        },
        index=X.index,
    )
    assert_frame_equal(X_trafo_ctg, expected_ctg)


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_return_indicator(calendar):
    """Test return_indicator param."""
    X = pd.DataFrame(
        {"values": np.arange(1, 6)},
        index=pd.date_range("2022-05-01", periods=5, freq="D"),
    )
    trafo_id = HolidayFeatures(
        calendar=calendar, return_indicator=True, return_dummies=False
    )
    X_trafo_id = trafo_id.fit_transform(X).astype(np.int32)
    expected_id = pd.DataFrame({"is_holiday": np.int32([0, 1, 0, 0, 0])}, index=X.index)
    assert_frame_equal(X_trafo_id, expected_id)


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_keep_original_column(calendar):
    """Tests keep_original_column param."""
    X = pd.DataFrame(
        {"values": np.arange(1, 6)},
        index=pd.date_range("2022-05-01", periods=5, freq="D"),
    )
    trafo_koc = HolidayFeatures(
        calendar=calendar,
        return_indicator=True,
        keep_original_columns=True,
        return_dummies=False,
    )
    X_trafo_koc = trafo_koc.fit_transform(X).astype(np.int32)
    expected_koc = pd.DataFrame(
        {
            "values": np.arange(1, 6).astype(np.int32),
            "is_holiday": np.int32([0, 1, 0, 0, 0]),
        },
        index=X.index,
    )
    assert_frame_equal(X_trafo_koc, expected_koc)


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_include_weekend(calendar):
    """Tests include_weekend param."""
    X = pd.DataFrame(
        {"values": np.arange(1, 6)},
        index=pd.date_range("2022-05-01", periods=5, freq="D"),
    )
    trafo_iw = HolidayFeatures(
        calendar=calendar,
        return_indicator=True,
        include_weekend=True,
        return_dummies=False,
    )
    X_trafo_iw = trafo_iw.fit_transform(X).astype(np.int32)
    expected_iw = pd.DataFrame({"is_holiday": np.int32([1, 1, 0, 0, 0])}, index=X.index)
    assert_frame_equal(X_trafo_iw, expected_iw)


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_holiday_not_in_window():
    calendar = {
        date(2024, 12, 25): "Natal",
        date(2023, 12, 25): "Natal",
        date(2022, 12, 25): "Natal",
        date(2021, 12, 25): "Natal",
        date(2024, 11, 29): "Black Friday",
        date(2023, 11, 29): "Black Friday",
        date(2022, 11, 29): "Black Friday",
        date(2021, 11, 29): "Black Friday",
        date(2024, 10, 1): "Dia Internacional do Café",
        date(2023, 10, 1): "Dia Internacional do Café",
        date(2022, 10, 1): "Dia Internacional do Café",
        date(2021, 10, 1): "Dia Internacional do Café",
        date(2024, 5, 12): "Dia das Mães",
        date(2023, 5, 12): "Dia das Mães",
        date(2022, 5, 12): "Dia das Mães",
        date(2021, 5, 12): "Dia das Mães",
        date(2024, 8, 11): "Dia dos Pais",
        date(2023, 8, 11): "Dia dos Pais",
        date(2022, 8, 11): "Dia dos Pais",
        date(2021, 8, 11): "Dia dos Pais",
        date(2024, 3, 15): "Semana do Consumidor",
        date(2023, 3, 15): "Semana do Consumidor",
        date(2022, 3, 15): "Semana do Consumidor",
        date(2021, 3, 15): "Semana do Consumidor",
    }
    holiday_transformer = HolidayFeatures(
        calendar=calendar,
        holiday_windows={
            "Natal": (5, 2),
            "Black Friday": (5, 2),
            "Dia Internacional do Café": (5, 2),
            "Dia das Mães": (5, 2),
            "Dia dos Pais": (5, 2),
            "Semana do Consumidor": (0, 6),
        },
    )

    ix = pd.date_range("2022-01-01", end="2022-05-31")
    X = pd.Series(14, index=ix)
    X_transformed = holiday_transformer.fit_transform(X)
    assert X_transformed.shape[0] == X.shape[0]


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_period_index(calendar):
    X_period = pd.DataFrame(
        {"values": np.arange(1, 6)},
        index=pd.period_range("2022-05-01", periods=5, freq="D"),
    )
    X_datetime = pd.DataFrame(
        {"values": np.arange(1, 6)},
        index=pd.date_range("2022-05-01", periods=5, freq="D"),
    )
    transformer = HolidayFeatures(
        calendar=calendar, return_indicator=True, return_dummies=True
    )
    result_period = transformer.fit_transform(X_period)
    result_datetime = transformer.fit_transform(X_datetime)
    assert isinstance(result_period.index, pd.PeriodIndex)
    result_datetime.index = pd.PeriodIndex(result_datetime.index, freq="D")
    assert_frame_equal(result_period, result_datetime)


def _holiday_labels(transformer, start, end):
    """Return categorical holiday labels as a dict of date strings to labels."""
    X = pd.DataFrame(
        {"values": 0.0}, index=pd.date_range(start=start, end=end, freq="D")
    )
    X_trafo = transformer.fit_transform(X)
    labels = X_trafo["holiday"].astype(str)
    return dict(zip(labels.index.strftime("%Y-%m-%d"), labels))


CHRISTMAS_NEW_YEAR = {
    date(2025, 12, 25): "Christmas",
    date(2026, 1, 1): "New Year",
}


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_overlapping_windows_combine_labels():
    """Tests that overlapping holiday windows keep both holiday names."""
    transformer = HolidayFeatures(
        calendar=CHRISTMAS_NEW_YEAR,
        holiday_windows={"Christmas": (0, 4), "New Year": (4, 0)},
        return_categorical=True,
    )
    labels = _holiday_labels(transformer, "2025-12-26", "2025-12-31")
    assert labels == {
        "2025-12-26": "Christmas",
        "2025-12-27": "Christmas",
        "2025-12-28": "Christmas, New Year",
        "2025-12-29": "Christmas, New Year",
        "2025-12-30": "New Year",
        "2025-12-31": "New Year",
    }

    X = pd.DataFrame(
        {"values": 0.0}, index=pd.date_range("2025-12-26", "2025-12-31", freq="D")
    )
    dummies = HolidayFeatures(
        calendar=CHRISTMAS_NEW_YEAR,
        holiday_windows={"Christmas": (0, 4), "New Year": (4, 0)},
    ).fit_transform(X)
    assert list(dummies.columns) == ["Christmas", "Christmas, New Year", "New Year"]
    assert dummies.sum().tolist() == [2, 2, 2]


@pytest.mark.parametrize(
    "holiday_windows",
    [
        {"Christmas": (2, 0), "Unknown Holiday": (1, 0)},
        {"Unknown Holiday": (1, 0), "Christmas": (2, 0)},
    ],
)
@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_unknown_holiday_in_windows_is_skipped(holiday_windows):
    """Tests that a window for a holiday not in the calendar has no effect."""
    transformer = HolidayFeatures(
        calendar=CHRISTMAS_NEW_YEAR,
        holiday_windows=holiday_windows,
        return_categorical=True,
    )
    with pytest.warns(UserWarning, match="Unknown Holiday"):
        labels = _holiday_labels(transformer, "2025-12-20", "2025-12-26")
    assert labels == {
        "2025-12-20": "no_holiday",
        "2025-12-21": "no_holiday",
        "2025-12-22": "no_holiday",
        "2025-12-23": "Christmas",
        "2025-12-24": "Christmas",
        "2025-12-25": "Christmas",
        "2025-12-26": "no_holiday",
    }


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_windows_raise_no_conflict_warning():
    """Tests that windows without conflicts do not raise warnings.

    This includes the holiday itself, and a holiday spanning consecutive days.
    """
    calendar = {
        date(2025, 12, 25): "Christmas",
        date(2026, 2, 16): "Carnival",
        date(2026, 2, 17): "Carnival",
    }
    transformer = HolidayFeatures(
        calendar=calendar,
        holiday_windows={"Christmas": (1, 1), "Carnival": (1, 1)},
        return_categorical=True,
    )
    X = pd.DataFrame(
        {"values": 0.0}, index=pd.date_range("2025-12-01", "2026-02-28", freq="D")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X_trafo = transformer.fit_transform(X)
    counts = X_trafo["holiday"].value_counts()
    assert counts["Christmas"] == 3
    assert counts["Carnival"] == 4


@pytest.mark.parametrize(
    "start, end, expected",
    [
        # holiday after the end of the index
        (
            "2025-12-20",
            "2025-12-24",
            ["no_holiday", "no_holiday", "Christmas", "Christmas", "Christmas"],
        ),
        # holiday before the start of the index
        (
            "2025-12-26",
            "2025-12-30",
            ["Christmas", "Christmas", "Christmas", "no_holiday", "no_holiday"],
        ),
    ],
)
@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_window_of_holiday_outside_index(start, end, expected):
    """Tests that windows of holidays just outside the time index are applied."""
    transformer = HolidayFeatures(
        calendar={date(2025, 12, 25): "Christmas"},
        holiday_windows={"Christmas": (3, 3)},
        return_categorical=True,
    )
    labels = _holiday_labels(transformer, start, end)
    assert list(labels.values()) == expected


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_window_of_holiday_outside_index_with_holidays_calendar():
    """Tests windows of holidays outside the index for a lazy HolidayBase calendar.

    HolidayBase objects only populate years on lookup, so the year of the
    holiday outside the index is not populated when the transformer is called.
    """
    from holidays import country_holidays

    transformer = HolidayFeatures(
        calendar=country_holidays(country="GB"),
        holiday_windows={"New Year's Day": (2, 0)},
        return_categorical=True,
    )
    labels = _holiday_labels(transformer, "2022-12-27", "2022-12-31")
    assert labels["2022-12-29"] == "no_holiday"
    assert labels["2022-12-30"] == "New Year's Day"
    assert labels["2022-12-31"] == "New Year's Day"


@pytest.mark.parametrize(
    "calendar, start, end, bridge_day",
    [
        # holiday on Tuesday after the end of the index, bridge day on Monday
        ({date(2026, 5, 5): "Holiday"}, "2026-04-30", "2026-05-04", "2026-05-04"),
        # holiday on Thursday before the start of the index, bridge day on Friday
        ({date(2026, 5, 14): "Holiday"}, "2026-05-15", "2026-05-19", "2026-05-15"),
    ],
)
@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_bridge_day_of_holiday_outside_index(calendar, start, end, bridge_day):
    """Tests that bridge days of holidays just outside the time index are applied."""
    transformer = HolidayFeatures(
        calendar=calendar,
        include_bridge_days=True,
        return_categorical=True,
    )
    labels = _holiday_labels(transformer, start, end)
    assert labels == {**dict.fromkeys(labels, "no_holiday"), bridge_day: "Holiday"}


@pytest.mark.skipif(
    not run_test_for_class(HolidayFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_negative_window_raises():
    """Tests that negative days in holiday windows raise an error."""
    transformer = HolidayFeatures(
        calendar=CHRISTMAS_NEW_YEAR,
        holiday_windows={"Christmas": (-1, 3)},
    )
    X = pd.DataFrame(
        {"values": 0.0}, index=pd.date_range("2025-12-20", "2025-12-31", freq="D")
    )
    with pytest.raises(ValueError, match="non-negative"):
        transformer.fit_transform(X)
