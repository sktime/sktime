#%%
from sktime.utils._testing.series import _make_series
from sktime.forecasting.dynamic_factor import DynamicFactor
from sktime.datasets import load_macroeconomic
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd
import numpy as np
# Load the longley dataset
COVERAGES = [0.95, 0.99]
df = load_macroeconomic()
y_train = df.loc[:'2007Q4', ['realgdp', 'realcons', 'realinv']].copy()
y_test = df.loc['2008Q1':, ['realgdp', 'realcons', 'realinv']].copy()
X_train = df.loc[:'2007Q4', ['unemp', 'pop', 'infl','cpi']].copy()
X_test = df.loc['2008Q1':, ['unemp', 'pop', 'infl','cpi']].copy()

# Create a forecasting horizon
fh_out = ForecastingHorizon([2, 3,4,5,6,7], is_relative=True)
# Fit the Dynamic Factor model
forecaster = DynamicFactor(enforce_stationarity=False, k_factors=2, factor_order=1)

forecaster.fit(y_train, X=X_train)
y_pred_out = forecaster.predict(fh=fh_out, X=X_test)

# Create a forecasting horizon
fh_in = ForecastingHorizon([2, 3,4,5,6,7], is_relative=False)
y_pred_in = forecaster.predict(fh=fh_in, X=X_train)
from statsmodels.tsa.statespace.dynamic_factor import (
            DynamicFactor as _DynamicFactor,
        )
mdl = _DynamicFactor(endog=y_train, k_factors=2, factor_order=1, exog=X_train, enforce_stationarity=False)
fitted = mdl.fit(disp=False)
y_pred_in_stats = fitted.predict(start=2, end=7, exog=X_train)

from pandas.testing import assert_frame_equal
assert_frame_equal(y_pred_in, y_pred_in_stats)
# %%
