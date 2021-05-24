# -*- coding: utf-8 -*-
from sktime.forecasting.arima import ARIMA
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
import numpy as np
import pandas as pd


from sktime.forecasting.compose import TransformedTargetForecaster

n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]

estimator = ARIMA()
steps = [
    ("deseasonalise", Deseasonalizer()),
    ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
    ("estimator", estimator),
]
s = TransformedTargetForecaster(steps)
s.fit(y_train)
result = [1, 5, 3]
np.testing.assert_allclose(result, s.predict(fh=[1, 3, 4]))

# d = STLForecaster(estimator, steps)
# d.fit(y_train)
