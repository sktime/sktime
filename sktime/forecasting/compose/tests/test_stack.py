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
    return pd.DataFrame({"feat": np.arange(len(y))}, index=y.index)


def _future_index(y, steps):
    last_idx = y.index[-1]
    return pd.period_range(start=last_idx + 1, periods=steps, freq=y.index.freq)


def _future_exog(y, steps):
    return pd.DataFrame(
        {"feat": np.arange(len(y), len(y) + steps)}, index=_future_index(y, steps)
    )


def test_stacking_forecaster_uses_exogenous_when_enabled():
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

    X_pred = _future_exog(y_train, len(fh))
    y_pred = sf.predict(fh=fh, X=X_pred)

    assert len(y_pred) == len(fh)
    # ensure predicted index aligns to absolute fh
    expected_index = _future_index(y_train, len(fh))
    pd.testing.assert_index_equal(y_pred.index, expected_index)


def test_stacking_forecaster_raises_without_X_when_required():
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

    with pytest.raises(ValueError):
        sf.predict(fh=fh)


def test_stacking_forecaster_raises_when_fitting_without_X():
    y = load_airline()
    forecasters = [("n1", NaiveForecaster()), ("n2", NaiveForecaster())]
    fh = [1, 2, 3]

    sf = StackingForecaster(forecasters=forecasters, use_exogenous_in_regressor=True)
    with pytest.raises(ValueError):
        sf.fit(y=y, fh=fh)
