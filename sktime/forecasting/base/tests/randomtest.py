# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for random_state Parameter."""

__author__ = ["Ris-Bali"]
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from sktime.datasets import load_airline
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.structural import UnobservedComponents
from sktime.forecasting.var import VAR
from sktime.utils._testing.forecasting import make_forecasting_problem

fh = np.arange(1, 5)

y = load_airline()
y_1 = make_forecasting_problem(n_columns=3)


def test_random_state():
    """Function for testing models with random_state parameter."""
    models = []
    models.append(AutoETS)
    models.append(ExponentialSmoothing)
    models.append(SARIMAX)
    models.append(UnobservedComponents)

    for model in models:
        forecaster = model(random_state=0)
        forecaster.fit(y=y, fh=fh)
        y_pred = forecaster.predict()
        forecaster.fit(y=y, fh=fh)
        y_pred_1 = forecaster.predict()
        assert_series_equal(y_pred, y_pred_1)

    forecaster = VAR(random_state=0)
    forecaster.fit(y=y_1, fh=fh)
    y_pred = forecaster.predict()
    forecaster.fit(y=y_1, fh=fh)
    y_pred_1 = forecaster.predict()
    assert_frame_equal(y_pred, y_pred_1)
