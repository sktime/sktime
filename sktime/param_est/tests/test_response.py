from sktime.param_est.impulse import ImpulseResponseFunction
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX as statvar
from sktime.forecasting.varmax import VARMAX as skvar

data = np.random.randn(100, 2)

st_model = statvar(data, order=(1, 0))
fitted_model = st_model.fit()
eg = fitted_model.impulse_responses()
print(eg)


sk_model = skvar(order=(1, 0))
fitted_model2 = sk_model.fit(data)
res = ImpulseResponseFunction(sk_model).get_irf_from_sktime()
print(res)