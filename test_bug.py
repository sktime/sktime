from sktime.forecasting.dynamic_factor import DynamicFactor
from sktime.utils._testing.series import _make_series

y = _make_series(n_columns=2)

model = DynamicFactor()
model.fit(y)

print("Before calling diagnostics")
model.plot_diagnostics()
print("After calling diagnostics")
