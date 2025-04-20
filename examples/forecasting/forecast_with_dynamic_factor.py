#%%
from sktime.utils._testing.series import _make_series
from sktime.forecasting.dynamic_factor import DynamicFactor
import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm

y = _make_series(n_columns=4)
forecaster = DynamicFactor()  
forecaster.fit(y)  
DynamicFactor(...)
y_pred = forecaster.predict(fh=[1,2,3])  
# %%
print("hello world")
# %%
