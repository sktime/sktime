# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests the conformal interval wrapper."""

__author__ = ["fkiraly"]

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.datatypes import convert_to, scitype_to_mtype
from sktime.forecasting.conformal import ConformalIntervals
from sktime.forecasting.naive import NaiveForecaster, NaiveVariance

INTERVAL_WRAPPERS = [ConformalIntervals, NaiveVariance]
MTYPES_SERIES = scitype_to_mtype("Series", softdeps="present")


@pytest.mark.parametrize("mtype", MTYPES_SERIES)
@pytest.mark.parametrize("override_y_mtype", [True, False])
@pytest.mark.parametrize("wrapper", INTERVAL_WRAPPERS)
def test_wrapper_series_mtype(wrapper, override_y_mtype, mtype):
    """Test that interval wrappers behave nicely with different internal y_mtypes.

    The wrappers require y to be pd.Series, and the internal estimator can have
    a different internal mtype.

    We test all interval wrappers in sktime (wrapper).

    We test once with an internal forecaster that needs pd.DataFrame conversion,
    and one that accepts pd.Series.
    We do this with a trick: the vanilla NaiveForecaster can accept both; we mimick a
    "pd.DataFrame only" forecaster by restricting its y_inner_mtype tag to pd.Series.
    """
    y = load_airline()
    y = convert_to(y, to_type=mtype)

    f = NaiveForecaster()

    if override_y_mtype:
        f.set_tags(**{"y_inner_mtype": "pd.DataFrame"})

    interval_forecaster = wrapper(f)
    interval_forecaster.fit(y, fh=[1, 2, 3])
    pred_int = interval_forecaster.predict_interval()

    assert isinstance(pred_int, pd.DataFrame)
    assert len(pred_int) == 3

    pred_var = interval_forecaster.predict_var()

    assert isinstance(pred_var, pd.DataFrame)
    assert len(pred_var) == 3
