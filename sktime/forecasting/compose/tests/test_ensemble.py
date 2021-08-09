#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Guzal Bulatova"]

import numpy as np
import pandas as pd
import pytest

from scipy.stats import gmean
from sktime.forecasting.compose import EnsembleForecaster
from sktime.tests._config import FORECASTERS
from sktime.utils._testing.forecasting import make_forecasting_problem


def test_avg_mean(forecasters=FORECASTERS):
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
def test_aggregation_unweighted(aggfunc):
    """Assert aggfunc returns the correct values."""
    y = make_forecasting_problem()
    forecaster = EnsembleForecaster(forecasters=FORECASTERS, aggfunc=aggfunc)
    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    predictions = []
    if aggfunc == "gmean":
        aggfunc = gmean
    for _, forecaster in FORECASTERS:
        f = forecaster
        f.fit(y)
        f_pred = f.predict(fh=[1, 2, 3])
        predictions.append(f_pred)
    predictions = pd.DataFrame(predictions)
    expected_pred = predictions.apply(func=aggfunc, axis=0)

    pd.testing.assert_series_equal(actual_pred, expected_pred)


@pytest.mark.parametrize("aggfunc", ["median", "gmean"])
@pytest.mark.parametrize("weights", [[1.44, 1.2]])
def test_aggregation_weighted(aggfunc, weights):
    """Assert aggfunc returns the correct values."""
    y = make_forecasting_problem()
    forecaster = EnsembleForecaster(
        forecasters=FORECASTERS, aggfunc=aggfunc, weights=weights
    )
    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    predictions = []
    for _, forecaster in FORECASTERS:
        f = forecaster
        f.fit(y)
        f_pred = f.predict(fh=[1, 2, 3])
        predictions.append(f_pred)

    predictions = pd.DataFrame(predictions)
    if aggfunc == "median":
        func = np.average
    else:
        func = gmean
    expected_pred = predictions.apply(func=func, axis=0, weights=weights)

    pd.testing.assert_series_equal(actual_pred, expected_pred)


@pytest.mark.parametrize("aggfunc", ["miin", "maximum", ""])
def test_invalid_aggfuncs(aggfunc):
    """Check if invalid aggregation functions return Error."""
    y = make_forecasting_problem()
    forecaster = EnsembleForecaster(forecasters=FORECASTERS, aggfunc=aggfunc)
    forecaster.fit(y, fh=[1, 2])
    with pytest.raises(ValueError, match=r"not recognized"):
        forecaster.predict()
