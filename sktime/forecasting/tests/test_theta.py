# -*- coding: utf-8 -*-
__author__ = ["@big-o"]

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.theta import ThetaForecaster
from sktime.utils.validation.forecasting import check_fh


def test_predictive_performance_on_airline():
    y = np.log1p(load_airline())
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(len(y_test)) + 1

    f = ThetaForecaster(sp=12)
    f.fit(y_train)
    y_pred = f.predict(fh=fh)

    # Performance on this particular dataset should be reasonably good.
    np.testing.assert_allclose(y_pred, y_test, rtol=0.05)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_pred_errors_against_y_test(fh):
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    f = ThetaForecaster()
    f.fit(y_train, fh=fh)

    y_pred = f.predict(return_pred_int=False)

    intervals = f.compute_pred_int(y_pred, [0.1])

    y_test = y_test.iloc[check_fh(fh) - 1]

    # Performance should be good enough that all point forecasts lie within the
    # prediction intervals.
    for ints in intervals:
        assert np.all(y_test > ints["lower"])
        assert np.all(y_test < ints["upper"])


def test_forecaster_with_initial_level():
    y = np.log1p(load_airline())
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(len(y_test)) + 1

    f = ThetaForecaster(initial_level=0.1, sp=12)
    f.fit(y_train)
    y_pred = f.predict(fh=fh)

    np.testing.assert_allclose(y_pred, y_test, rtol=0.05)
