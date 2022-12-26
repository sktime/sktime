#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Tests for ClearSky transformer."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_solar
from sktime.transformations.series.clear_sky import ClearSky
from sktime.utils.validation._dependencies import _check_soft_dependencies

output_chk = [0.0, 0.0, 0.901, 0.739, 0.618, 0.0]


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for statsmodels not available",
)
def test_clearsky_trafo_vals():
    """Tests clear sky trafo with and without missing values and period index."""
    y = load_solar(api_version=None)
    # only take every 4H for quickness
    y = y.asfreq("4H")
    cs_model = ClearSky()
    y_trafo = cs_model.fit_transform(y)

    msg = "ClearSky not transforming values consistently with stored values."
    assert np.all(output_chk == y_trafo[0:6].round(3).tolist()), msg

    y_missing = y.copy()
    y_missing.iloc[6:12] = np.nan
    cs_model_missing = ClearSky()
    y_trafo_missing = cs_model_missing.fit_transform(y_missing)

    msg = "ClearSky transformer not returning correct number of values."
    assert y.count() == y_trafo.count(), msg
    assert y_missing.count() == y_trafo_missing.count(), msg

    y_period = y.copy()
    y_period.index = y_period.index.to_period()
    cs_model_period = ClearSky()
    y_trafo_period = cs_model_period.fit_transform(y_period)

    msg = "PeriodIndex and DatetimeIndex returning different values"
    assert np.all(y_trafo_period.values == y_trafo.values), msg


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for statsmodels not available",
)
def test_clearsky_trafo_range_exception():
    """Tests clear sky trafo exception with range index."""
    y = load_solar(api_version=None)

    # range index should not work
    y = y.reset_index(drop=True)
    cs_model = ClearSky()
    with pytest.raises(ValueError):
        cs_model.fit_transform(y)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for statsmodels not available",
)
def test_clearsky_trafo_nofreq_exception():
    """Tests clear sky trafo exception with no set/inferrable freq in index."""
    y = load_solar(api_version=None)

    # no set or inferrable frequency should not work
    y = y.drop(pd.to_datetime("2021-05-01 00:30:00", utc=True))
    cs_model = ClearSky()
    with pytest.raises(ValueError):
        cs_model.fit_transform(y)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for statsmodels not available",
)
def test_clearsky_trafo_grdaily_exception():
    """Tests clear sky trafo exception with greater than daily freq in index."""
    y = load_solar(api_version=None)

    # gr daily frequency should not work
    y = y.asfreq("2D")
    cs_model = ClearSky()
    with pytest.raises(ValueError):
        cs_model.fit_transform(y)
