#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of ColumnEnsembleForecaster functionality."""

__author__ = ["GuzalBulatova"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster


@pytest.mark.parametrize(
    "forecasters",
    [
        [
            ("trend", PolynomialTrendForecaster(), 0),
            ("naive", NaiveForecaster(), 1),
            ("ses", ExponentialSmoothing(), 2),
        ]
    ],
)
@pytest.mark.parametrize(
    "fh", [(np.arange(1, 11)), (np.arange(1, 33)), (np.arange(1, 3))]
)
def test_column_ensemble_shape(forecasters, fh):
    """Check the shape of the returned prediction."""
    y = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list("ABC"))
    forecaster = ColumnEnsembleForecaster(forecasters)
    forecaster.fit(y, fh=fh)
    actual = forecaster.predict()
    assert actual.shape == (len(fh), y.shape[1])


@pytest.mark.parametrize(
    "forecasters",
    [
        [
            ("trend", PolynomialTrendForecaster(), 0),
            ("naive1", NaiveForecaster(), 1),
            ("naive", NaiveForecaster(), 0),
        ],
        [("trend", PolynomialTrendForecaster(), 0), ("naive", NaiveForecaster(), 1)],
    ],
)
def test_invalid_forecasters_indices(forecasters):
    """Check if invalid forecaster indices return correct Error."""
    y = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list("ABC"))
    forecaster = ColumnEnsembleForecaster(forecasters=forecasters)
    with pytest.raises(ValueError, match=r"estimator per column"):
        forecaster.fit(y, fh=[1, 2])
