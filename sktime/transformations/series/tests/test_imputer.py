#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Imputer functionality."""

__author__ = ["aiwalter"]
__all__ = []

import numpy as np
import pytest

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.transformations.series.impute import Imputer
from sktime.utils._testing.forecasting import make_forecasting_problem

y, X = make_forecasting_problem(make_X=True)

X.iloc[3, 0] = np.nan
X.iloc[3, 1] = np.nan
X.iloc[0, 1] = np.nan
X.iloc[-1, 1] = np.nan

y.iloc[3] = np.nan
y.iloc[0] = np.nan
y.iloc[-1] = np.nan


@pytest.mark.parametrize("Z", [y, X])
@pytest.mark.parametrize(
    "method",
    [
        "drift",
        "linear",
        "nearest",
        "constant",
        "mean",
        "median",
        "backfill",
        "pad",
        "random",
        "forecaster",
    ],
)
def test_imputer(method, Z):
    """Test univariate and multivariate Imputer with all methods."""
    forecaster = ExponentialSmoothing() if method == "forecaster" else None
    value = 3 if method == "constant" else None
    t = Imputer(method=method, forecaster=forecaster, value=value)
    y_hat = t.fit_transform(Z)
    assert not y_hat.isnull().to_numpy().any()
