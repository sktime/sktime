#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for HolidayFeatures functionality."""

__author__ = ["VyomkeshVyas", "fnhirwa"]

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.holiday._holidayfeats import HolidayFeatures


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
