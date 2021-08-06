#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Guzal Bulatova"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster


@pytest.mark.parametrize(
    "forecasters",
    [
        [("trend", PolynomialTrendForecaster()), ("naive", NaiveForecaster())],
        [("trend", PolynomialTrendForecaster()), ("ses", ExponentialSmoothing())],
    ],
)
def test_avg_mean(forecasters):
    """Assert `mean` aggfunc returns the same values as `avg` with equal weights."""
    y = pd.DataFrame(np.random.randint(0, 100, size=(100,)))
    forecaster = EnsembleForecaster(forecasters=forecasters)
    forecaster.fit(y, fh=[1, 2, 3])
    mean_pred = forecaster.predict()

    forecaster_1 = EnsembleForecaster(
        forecasters=forecasters, aggfunc="mean", weights=[1, 1]
    )
    forecaster_1.fit(y, fh=[1, 2, 3])
    avg_pred = forecaster_1.predict()

    pd.testing.assert_series_equal(mean_pred, avg_pred)


# @pytest.mark.parametrize("aggfunc", ["min", "max", ""])
# @pytest.mark.parametrize(
#     "forecasters",
#     [[("trend", PolynomialTrendForecaster(), 0), ("naive", NaiveForecaster(), 1)]],
# )
# def test_invalid_aggfuncs(forecasters, aggfunc):
#     """Check if invalid aggregation functions return Error."""
#     y = pd.DataFrame(np.random.randint(0, 100, size=(100,)))
#     forecaster = EnsembleForecaster(forecasters=forecasters, aggfunc=aggfunc)
#     forecaster.fit(y, fh=[1, 2])
#     with pytest.raises(ValueError, match=r"not recognized"):
#         forecaster.predict()
