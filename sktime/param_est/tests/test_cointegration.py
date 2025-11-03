
from sktime.datasets import load_airline
from sktime.param_est.cointegration import JohansenCointegration



X = load_airline()
coint_est = JohansenCointegration()
coint_est.fit(X)

print(coint_est.trace_stat)
print(coint_est.trace_stat_crit_vals)