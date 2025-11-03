
from sktime.datasets import load_airline
from sktime.param_est.cointegration import JohansenCointegration
import pandas as pd


X = load_airline()
X2 = X.shift(1).bfill()
df = pd.DataFrame({"X":X, "X2": X2})
print(df)
coint_est = JohansenCointegration()
coint_est.fit(df)

print(coint_est.trace_stat)
print(coint_est.trace_stat_crit_vals)