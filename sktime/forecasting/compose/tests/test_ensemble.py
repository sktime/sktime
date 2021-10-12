#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of EnsembleForecaster functionality."""

__author__ = ["GuzalBulatova"]

import numpy as np
import pandas as pd
import pytest
import sys

from scipy.stats import gmean
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster

from sktime.utils._testing.forecasting import make_forecasting_problem


@pytest.mark.parametrize(
    "forecasters",
    [
        [("trend", PolynomialTrendForecaster()), ("naive", NaiveForecaster())],
        [("trend", PolynomialTrendForecaster()), ("ses", ExponentialSmoothing())],
    ],
)
def test_avg_mean(forecasters):
    """Assert `mean` aggfunc returns the same values as `average` with equal weights."""
    y = make_forecasting_problem()
    forecaster = EnsembleForecaster(forecasters)
    forecaster.fit(y, fh=[1, 2, 3])
    mean_pred = forecaster.predict()

    forecaster_1 = EnsembleForecaster(forecasters, aggfunc="mean", weights=[1, 1])
    forecaster_1.fit(y, fh=[1, 2, 3])
    avg_pred = forecaster_1.predict()

    pd.testing.assert_series_equal(mean_pred, avg_pred)


@pytest.mark.parametrize("aggfunc", ["median", "mean", "min", "max", "gmean"])
@pytest.mark.parametrize(
    "forecasters",
    [
        [("trend", PolynomialTrendForecaster()), ("naive", NaiveForecaster())],
    ],
)
def test_aggregation_unweighted(forecasters, aggfunc):
    """Assert aggfunc returns the correct values."""
    y = make_forecasting_problem()
    forecaster = EnsembleForecaster(forecasters=forecasters, aggfunc=aggfunc)
    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    predictions = []
    if aggfunc == "gmean":
        aggfunc = gmean
    for _, forecaster in forecasters:
        f = forecaster
        f.fit(y)
        f_pred = f.predict(fh=[1, 2, 3])
        predictions.append(f_pred)
    predictions = pd.DataFrame(predictions)
    expected_pred = predictions.apply(func=aggfunc, axis=0)

    pd.testing.assert_series_equal(actual_pred, expected_pred)


@pytest.mark.parametrize("aggfunc", ["mean", "gmean"])
@pytest.mark.parametrize("weights", [[1.44, 1.2]])
@pytest.mark.parametrize(
    "forecasters",
    [
        [("trend", PolynomialTrendForecaster()), ("naive", NaiveForecaster())],
    ],
)
@pytest.mark.skipif(sys.version_info < (3, 7), reason="requires python3.7 or higher")
def test_aggregation_weighted(forecasters, aggfunc, weights):
    """Assert weighted aggfunc returns the correct values."""
    y = make_forecasting_problem()
    forecaster = EnsembleForecaster(
        forecasters=forecasters, aggfunc=aggfunc, weights=weights
    )
    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    predictions = []
    for _, forecaster in forecasters:
        f = forecaster
        f.fit(y)
        f_pred = f.predict(fh=[1, 2, 3])
        predictions.append(f_pred)

    predictions = pd.DataFrame(predictions)
    if aggfunc == "mean":
        func = np.average
    else:
        func = gmean
    expected_pred = predictions.apply(func=func, axis=0, weights=weights)

    pd.testing.assert_series_equal(actual_pred, expected_pred)


@pytest.mark.parametrize("aggfunc", ["miin", "maximum", ""])
@pytest.mark.parametrize(
    "forecasters",
    [[("trend", PolynomialTrendForecaster()), ("naive", NaiveForecaster())]],
)
def test_invalid_aggfuncs(forecasters, aggfunc):
    """Check if invalid aggregation functions return Error."""
    y = make_forecasting_problem()
    forecaster = EnsembleForecaster(forecasters=forecasters, aggfunc=aggfunc)
    forecaster.fit(y, fh=[1, 2])
    with pytest.raises(ValueError, match=r"not recognized"):
        forecaster.predict()
