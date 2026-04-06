import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor as statsdyn

from sktime.datasets import load_airline
from sktime.forecasting.dynamic_factor import DynamicFactor as skdyn
from sktime.param_est.impulse import ImpulseResponseFunction

X = load_airline()
X2 = X.shift(1).bfill()
df = pd.DataFrame({"X": X, "X2": X2})
data = np.random.randn(100, 2)
st_model = statsdyn(
    df, k_factors=1, factor_order=2, error_order=2, enforce_stationarity=False
)
fitted_model = st_model.fit()
stats_res = fitted_model.impulse_responses(steps=3)
print(stats_res)

sk_model = skdyn(
    k_factors=1, factor_order=2, error_order=2, enforce_stationarity=False
).fit(df)
sktime_res = ImpulseResponseFunction(sk_model, steps=3)
sktime_res.fit(df)
print(sktime_res.get_fitted_params()["irf"])
print("#####################")
