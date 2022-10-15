#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Detrender transformer tests."""

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.detrend import Detrender


def test_detrender_univariate():
    """Checks detrended time series given naive(mean) forecaster returns y - mean(y)."""
    y = pd.Series(np.arange(20) * 0.5) + np.random.normal(0, 1, size=20)
    forecaster = NaiveForecaster(strategy="mean")
    transformer = Detrender(forecaster)
    transformer.fit(y)

    # check residuals
    result = transformer.transform(y)
    expected = y - y.mean()
    np.testing.assert_array_almost_equal(result, expected)


def test_detrender_multivariate():
    """Checks detrended multivariate X given naive(mean) forecaster."""
    y = pd.DataFrame(
        {
            "a": np.arange(20) + np.random.normal(0, 1, size=20),
            "b": np.arange(20) + np.random.normal(0, 2, size=20),
            "c": np.arange(20) + np.random.normal(0, 3, size=20),
        }
    )
    forecaster = NaiveForecaster(strategy="mean")
    transformer = Detrender(forecaster)
    transformer.fit(y)

    # check residuals
    result = transformer.transform(y)
    expected = y - y.mean()
    np.testing.assert_array_almost_equal(result, expected)
