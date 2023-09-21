#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Imputer functionality."""

__author__ = ["aiwalter"]
__all__ = []

import numpy as np
import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.impute import Imputer
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.hierarchical import _make_hierarchical

y, X = make_forecasting_problem(make_X=True)

X.iloc[3, 0] = np.nan
X.iloc[3, 1] = np.nan
X.iloc[0, 1] = np.nan
X.iloc[-1, 1] = np.nan

y.iloc[3] = np.nan
y.iloc[0] = np.nan
y.iloc[-1] = np.nan

z = _make_hierarchical(hierarchy_levels=(3,), n_columns=3)

z.iloc[3] = np.nan
z.iloc[0] = np.nan
z.iloc[-1] = np.nan


@pytest.mark.parametrize("Z", [y, X, z])
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
    forecaster = NaiveForecaster() if method == "forecaster" else None
    value = 3 if method == "constant" else None
    t = Imputer(method=method, forecaster=forecaster, value=value)
    y_hat = t.fit_transform(Z)
    assert not y_hat.isnull().to_numpy().any()


def test_imputer_forecaster_y():
    """Test that forecaster imputer works with y.

    Failure case in bug #5284.
    """
    from sklearn.linear_model import LinearRegression

    from sktime.datasets import load_airline
    from sktime.forecasting.compose import YfromX

    X = load_airline()
    y = load_airline()

    model_reg = YfromX(LinearRegression())
    model_reg.fit(X, y)
    transformer = Imputer(method="forecaster", forecaster=model_reg)

    transformer.fit(X=X, y=y)
    transformer.transform(X=X, y=y)
