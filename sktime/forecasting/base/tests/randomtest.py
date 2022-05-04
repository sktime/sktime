# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for random_state Parameter."""

__author__ = ["Ris-Bali"]
import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "model",
    [AutoETS, ExponentialSmoothing, SARIMAX, UnobservedComponents, VAR],
)
def test_random_state(model):
    """Function to test random_state parameter."""
    obj = model.create_test_instance()
    if model == VAR:
        obj.fit(y=y_1, fh=fh)
        y = obj.predict()
        obj.fit(y=y_1, fh=fh)
        y1 = obj.predict()
    else:
        obj.fit(y=y, fh=fh)
        y = obj.predict()
        obj.fit(y=y, fh=fh)
        y1 = obj.predict()
    assert y == y1
