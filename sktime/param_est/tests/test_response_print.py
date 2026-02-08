from sktime.param_est.impulse import ImpulseResponseFunction
import numpy as np
from sktime.datasets import load_airline
import pandas as pd
#from statsmodels.tsa.statespace.varmax import VARMAX as statvar
#from sktime.forecasting.varmax import VARMAX as skvar

from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor as statvar
from sktime.forecasting.dynamic_factor import DynamicFactor as skvar


X = load_airline()
X2 = X.shift(1).bfill()
df = pd.DataFrame({"X": X, "X2": X2})

data = np.random.randn(100, 2)

st_model = statvar(df, k_factors=1, factor_order=2)
fitted_model = st_model.fit()
eg = fitted_model.impulse_responses()
print(eg)


sk_model = skvar(k_factors=1, factor_order=2).fit(df)
res = ImpulseResponseFunction(sk_model).fit(df)
print(res)
