from sktime.param_est.impulse import ImpulseResponseFunction
import numpy as np
#from statsmodels.tsa.statespace.varmax import VARMAX as statvar
#from sktime.forecasting.varmax import VARMAX as skvar

from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor as statvar
from sktime.forecasting.dynamic_factor import DynamicFactor as skvar

print(skvar.__name__)

data = np.random.randn(100, 2)

st_model = statvar(data, k_factors=1, factor_order=2)
fitted_model = st_model.fit()
eg = fitted_model.impulse_responses()
print(eg)


sk_model = skvar(k_factors=1, factor_order=2)
fitted_model2 = sk_model.fit(data)
res = ImpulseResponseFunction(sk_model).get_irf_from_sktime()
print(res)