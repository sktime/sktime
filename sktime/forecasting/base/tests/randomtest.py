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

forecaster = AutoETS(random_state=0)
forecaster.fit(y=y, fh=fh)
y_pred = forecaster.predict()
forecaster.fit(y=y, fh=fh)
y_pred1 = forecaster.predict()

assert_series_equal(y_pred, y_pred1)

forecaster = ExponentialSmoothing(randome_state=0)
forecaster.fit(y=y, fh=fh)
y_pred = forecaster.predict()
forecaster.fit(y=y, fh=fh)
y_pred1 = forecaster.predict()
assert_series_equal(y_pred, y_pred1)

forecaster = SARIMAX(random_state=0)
forecaster.fit(y=y, fh=fh)
y_pred = forecaster.predict()
forecaster.fit(y=y, fh=fh)
y_pred1 = forecaster.predict()
assert_series_equal(y_pred, y_pred1)

forecaster = UnobservedComponents(random_state=0)
forecaster.fit(y=y, fh=fh)
y_pred = forecaster.predict()
forecaster.fit(y=y, fh=fh)
y_pred1 = forecaster.predict()
assert_series_equal(y_pred, y_pred1)

y1 = make_forecasting_problem(n_columns=4)
forecaster = VAR(random_state=0)
forecaster.fit(y=y1, fh=fh)
y_pred = forecaster.predict()
forecaster.fit(y=y1, fh=fh)
y_pred1 = forecaster.predict()
assert_frame_equal(y_pred, y_pred1)
