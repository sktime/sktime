#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for HolidayFeatures functionality."""

__author__ = ["VyomkeshVyas"]

import numpy as np
import pandas as pd
import pytest
from holidays import CountryHoliday
from pandas.testing import assert_frame_equal

from sktime.transformations.series.holidays import HolidayFeatures
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("holidays", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_holiday_features_params():
    """Tests HolidayFeatures trafo params."""
    values = np.arange(1, 6)
    index = pd.date_range("2022-05-01", periods=5, freq="D")
    X = pd.DataFrame({"values": values}, index=index)
    calendar = CountryHoliday(country="GB")

    trafo_dm = HolidayFeatures(calendar=calendar, return_dummies=True)
    X_trafo_dm = trafo_dm.fit_transform(X)
    expected_dm = pd.DataFrame({"May Day": np.int32([0, 1, 0, 0, 0])}, index=index)
    assert_frame_equal(X_trafo_dm, expected_dm)

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
        index=index,
    )
    assert_frame_equal(X_trafo_ctg, expected_ctg)

    trafo_id = HolidayFeatures(
        calendar=calendar, return_indicator=True, return_dummies=False
    )
    X_trafo_id = trafo_id.fit_transform(X)
    expected_id = pd.DataFrame({"is_holiday": np.int32([0, 1, 0, 0, 0])}, index=index)
    assert_frame_equal(X_trafo_id, expected_id)

    trafo_koc = HolidayFeatures(
        calendar=calendar,
        return_indicator=True,
        return_dummies=False,
        keep_original_columns=True,
    )
    X_trafo_koc = trafo_koc.fit_transform(X)
    expected_koc = pd.DataFrame(
        {"values": values, "is_holiday": np.int32([0, 1, 0, 0, 0])}, index=index
    )
    assert_frame_equal(X_trafo_koc, expected_koc)
