# -*- coding: utf-8 -*-
"""Tests for ExponentialSmoothing.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
__author__ = ["Markus Löning", "@big-o"]
__all__ = ["test_set_params"]

import pytest
from numpy.testing import assert_array_equal

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils._testing.forecasting import make_forecasting_problem

# load test data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_set_params():
    """Test set parameters for ExponentialSmoothing."""
    params = {"trend": "additive"}

    f = ExponentialSmoothing(**params)
    fh = ForecastingHorizon(1, freq=y.index.freqstr)
    f.fit(y_train, fh=fh)
    expected = f.predict()

    f = ExponentialSmoothing()
    f.set_params(**params)
    f.fit(y_train, fh=fh)
    y_pred = f.predict()

    assert_array_equal(y_pred, expected)
