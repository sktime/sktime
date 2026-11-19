from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
import pandas as pd
import numpy as np

# load data
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=12)

print("=== SIMPLE MODEL TEST ===")
# simple model (no trend/seasonality)
f = ExponentialSmoothing()
f.fit(y_train)
f.update(y_test.iloc[:3])

f2 = ExponentialSmoothing()
f2.fit(pd.concat([y_train, y_test.iloc[:3]]))

pred1 = f.predict([1, 2, 3]).values
pred2 = f2.predict([1, 2, 3]).values

print("Update model:", pred1)
print("Full refit model:", pred2)

# exact match expected
print("Exact match:", np.allclose(pred1, pred2))


print("\n=== TREND + SEASONAL MODEL TEST ===")
# complex model
f = ExponentialSmoothing(trend="add", seasonal="add", sp=12)
f.fit(y_train)
f.update(y_test.iloc[:3])

f2 = ExponentialSmoothing(trend="add", seasonal="add", sp=12)
f2.fit(pd.concat([y_train, y_test.iloc[:3]]))

pred1 = f.predict([1, 2, 3]).values
pred2 = f2.predict([1, 2, 3]).values

print("Update model:", pred1)
print("Full refit model:", pred2)

# close match expected (not exact)
print("Close match:", np.allclose(pred1, pred2, rtol=1e-1))