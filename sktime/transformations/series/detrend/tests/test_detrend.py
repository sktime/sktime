#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Detrender transformer tests."""

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pandas as pd

from sktime.transformations.series.detrend import Detrender
from sktime.utils.estimators._forecasters import MockForecaster


def test_detrender_univariate():
    """Checks Detrender returns transformed y as residuals."""
    y = pd.Series(np.arange(20) * 0.5)
    prediction_constant = 10
    forecaster = MockForecaster(prediction_constant)
    transformer = Detrender(forecaster)
    transformer.fit(y)

    # check residuals
    result = transformer.transform(y)
    expected = y - prediction_constant
    pd.testing.assert_series_equal(result, expected)


def test_detrender_multivariate():
    """Checks Detrender returns transformed multivariate X as residuals."""
    y = pd.DataFrame(
        {
            "a": np.arange(20),
            "b": np.arange(20) * 0.5,
            "c": np.arange(20) * 2,
        }
    )
    prediction_constant = 10
    forecaster = MockForecaster(prediction_constant)
    transformer = Detrender(forecaster)
    transformer.fit(y)

    # check residuals
    result = transformer.transform(y)
    expected = y - prediction_constant
    pd.testing.assert_frame_equal(result, expected)
