# -*- coding: utf-8 -*-
__author__ = ["Markus Löning", "@big-o"]
__all__ = ["test_set_params"]

import pytest
from numpy.testing import assert_array_equal

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils._testing.forecasting import make_forecasting_problem

# load test data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_set_params():
    params = {"trend": "additive"}

    f = ExponentialSmoothing(**params)
    f.fit(y_train, fh=1)
    expected = f.predict()

    f = ExponentialSmoothing()
    f.set_params(**params)
    f.fit(y_train, fh=1)
    y_pred = f.predict()

    assert_array_equal(y_pred, expected)
