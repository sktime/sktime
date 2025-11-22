# <some parts will go into example in _johansen.py>
from sktime.datasets import load_airline
from sktime.param_est.cointegration import JohansenCointegration
import pandas as pd


X = load_airline()
X2 = X.shift(1).bfill()
df = pd.DataFrame({"X":X, "X2": X2})
coint_est = JohansenCointegration()
coint_est.fit(df)

print(coint_est.get_fitted_params()["ind"])

#print(coint_est.cvm_)
#print(coint_est.cvt_)
#print(coint_est.eig_)
#print(coint_est.evec_)
#print(coint_est.ind_)
#print(coint_est.lr1_)
#print(coint_est.lr2_)
#print(coint_est.max_eig_stat_)
#print(coint_est.max_eig_stat_crit_vals_)
#print(coint_est.meth_)
#print(coint_est.r0t_)
#print(coint_est.rkt_)
#print(coint_est.trace_stat_)
#print(coint_est.trace_stat_crit_vals_)
# </ some parts will go into example in _johansen.py>