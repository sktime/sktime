from sktime.param_est.impulse import ImpulseResponseFunction
import numpy as np
from sktime.datasets import load_airline
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX as statsmax
from sktime.forecasting.varmax import VARMAX as skmax

from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor as statsdyn
from sktime.forecasting.dynamic_factor import DynamicFactor as skdyn


X = load_airline()
X2 = X.shift(1).bfill()
df = pd.DataFrame({"X": X, "X2": X2})

data = np.random.randn(100, 2)

st_model = statsdyn(df, k_factors=1, factor_order=2)
fitted_model = st_model.fit()
stats_res = fitted_model.impulse_responses(orthogonalized=True)
print(stats_res)

sk_model = skdyn(k_factors=1, factor_order=2).fit(df)
sktime_res = ImpulseResponseFunction(sk_model, orthogonalized=True)
sktime_res.fit(df)
print(sktime_res.get_fitted_params()["irf"])

print("#####################")

X1 = load_airline().values.astype(float)
X1_stationary = np.diff(np.log(X1))
np.random.seed(42)
noise = np.random.normal(scale=0.05, size=len(X1_stationary))
X2_stationary = 0.6 * X1_stationary + 0.4 * np.roll(X1_stationary, 1) + noise
df2 = pd.DataFrame({
    "X1": X1_stationary[1:],
    "X2": X2_stationary[1:]
})
df2.index = pd.date_range("1949-02-01", periods=len(df2), freq="MS")

st_model2 = statsmax(df2, order=(1, 2), trend ="c")
fitted_model2 = st_model2.fit()
stats_res2 = fitted_model2.impulse_responses()
print(stats_res2)

sk_model2 = skmax(order=(1, 2), trend ="c").fit(df2)
sktime_res2 = ImpulseResponseFunction(sk_model2)
sktime_res2.fit(df2)
print(sktime_res2.get_fitted_params()["irf"])




