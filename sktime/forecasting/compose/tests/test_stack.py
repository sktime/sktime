#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for StackingForecaster exogenous handling."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.compose import StackingForecaster
from sktime.forecasting.naive import NaiveForecaster


def _make_exog(y):
    """Simple exogenous feature aligned with y index."""
    return pd.DataFrame({"feat": np.arange(len(y))}, index=y.index)


def _make_future_exog(y, fh):
    """Future exog aligned with fh (relative) for a monthly index."""
    last_idx = y.index[-1]
    # assume standard monthly index as in load_airline
    future_idx = pd.period_range(start=last_idx + 1, periods=len(fh), freq=y.index.freq)
    return pd.DataFrame({"feat": np.arange(len(y), len(y) + len(fh))}, index=future_idx)


def test_stacking_forecaster_uses_exogenous_when_enabled():
    """Meta-regressor should accept X when flag is enabled."""
    y = load_airline()
    X = _make_exog(y)
    fh = [1, 2, 3]

    y_train, X_train = y.iloc[:-3], X.iloc[:-3]
    forecasters = [
        ("n1", NaiveForecaster()),
        ("n2", NaiveForecaster(strategy="drift")),
    ]

    sf = StackingForecaster(forecasters=forecasters, use_exogenous_in_regressor=True)
    sf.fit(y=y_train, X=X_train, fh=fh)

    X_pred = _make_future_exog(y_train, fh)
    y_pred = sf.predict(fh=fh, X=X_pred)

    assert len(y_pred) == len(fh)
    pd.testing.assert_index_equal(y_pred.index, X_pred.index)


def test_stacking_forecaster_raises_without_X_in_fit_when_flag_enabled():
    """fit should raise if flag is on but X is not provided."""
    y = load_airline()
    fh = [1, 2, 3]
    forecasters = [("n1", NaiveForecaster()), ("n2", NaiveForecaster())]

    sf = StackingForecaster(forecasters=forecasters, use_exogenous_in_regressor=True)
    with pytest.raises(ValueError):
        sf.fit(y=y, fh=fh)


def test_stacking_forecaster_raises_without_X_in_predict_when_flag_enabled():
    """predict should raise if flag is on but X is not provided."""
    y = load_airline()
    X = _make_exog(y)
    fh = [1, 2, 3]

    y_train, X_train = y.iloc[:-3], X.iloc[:-3]
    forecasters = [("n1", NaiveForecaster()), ("n2", NaiveForecaster())]

    sf = StackingForecaster(forecasters=forecasters, use_exogenous_in_regressor=True)
    sf.fit(y=y_train, X=X_train, fh=fh)

    with pytest.raises(ValueError):
        sf.predict(fh=fh)
