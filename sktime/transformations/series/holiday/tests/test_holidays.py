#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for HolidayFeatures functionality."""

__author__ = ["VyomkeshVyas"]

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.transformations.series.holiday._holidayfeats import HolidayFeatures
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.fixture
def calendar():
    """Fixture for GB holidays."""
    from holidays import country_holidays

    return country_holidays(country="GB")


@pytest.mark.skipif(
    not _check_soft_dependencies("holidays", severity="none"),
    reason="skip test if required soft dependency not available",
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
    not _check_soft_dependencies("holidays", severity="none"),
    reason="skip test if required soft dependency not available",
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
    not _check_soft_dependencies("holidays", severity="none"),
    reason="skip test if required soft dependency not available",
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
    not _check_soft_dependencies("holidays", severity="none"),
    reason="skip test if required soft dependency not available",
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
    not _check_soft_dependencies("holidays", severity="none"),
    reason="skip test if required soft dependency not available",
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
