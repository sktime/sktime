#!/usr/bin/env python3 -u
"""Tests for window forecasters."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.registry import all_estimators
from sktime.split import temporal_train_test_split
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.series import _make_series

FH0 = 1

WINDOW_FORECASTERS = [
    forecaster
    for (name, forecaster) in all_estimators(estimator_types="forecaster")
    if issubclass(forecaster, _BaseWindowForecaster)
]

# testing data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.parametrize("Forecaster", WINDOW_FORECASTERS)
def test_last_window(Forecaster):
    """Test window forecaster common API points."""
    f = Forecaster.create_test_instance()
    n_columns = 1

    f = Forecaster.create_test_instance()
    y_train = _make_series(n_columns=n_columns)
    # passing the same fh to both fit and predict works
    f.fit(y_train, fh=FH0)

    actual, _ = f._get_last_window()
    expected = y_train.iloc[-f.window_length_ :]

    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == f.window_length_


def test_insample_and_outofsample_forecasting():
    n = 20
    df = pd.DataFrame(
        index=pd.period_range("2020-01-01", periods=n, freq="D"),
        data={"value": np.arange(n)},
    )

    model = NaiveForecaster(strategy="mean", window_length=7)
    model.fit(df)

    assert (model.predict(fh=[1, 2, 3]).values.flatten() == [16, 16, 16]).all()
    assert (model.predict(fh=[0]).values.flatten() == [15]).all()
    assert (model.predict(fh=[0, 1, 2, 3]).values.flatten() == [15, 16, 16, 16]).all()
