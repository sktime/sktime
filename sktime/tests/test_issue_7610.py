from sktime.datasets import load_forecastingdata
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
y, _ = load_forecastingdata("m1_monthly_dataset", return_type="pd_multiindex_hier")
# Sort indexes for safety
y = y.sort_index()


# Forecast horizon
fh = [1,2,3]

# Fit and predict
model = ExponentialSmoothing(trend="add")
model.fit(y)
y=model.predict(fh=fh) # the error is raised here
print(y)
