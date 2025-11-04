
# still needs a real test, this is the example usage to be used in the doc string

from sktime.datasets import load_airline
from sktime.param_est.cointegration import JohansenCointegration
import pandas as pd


X = load_airline()
X2 = X.shift(1).bfill()
df = pd.DataFrame({"X":X, "X2": X2})
coint_est = JohansenCointegration()
coint_est.fit(df)

print(coint_est.cvm)
print(coint_est.cvt)
print(coint_est.eig)
print(coint_est.evec)
print(coint_est.ind)
print(coint_est.lr1)
print(coint_est.lr2)
print(coint_est.max_eig_stat)
print(coint_est.max_eig_stat_crit_vals)
print(coint_est.meth)
print(coint_est.r0t)
print(coint_est.rkt)
print(coint_est.trace_stat)
print(coint_est.trace_stat_crit_vals)